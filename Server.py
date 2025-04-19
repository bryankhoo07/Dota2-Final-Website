from flask import Flask, request,jsonify
from flask_cors import CORS
import torch
import pickle
import numpy as np
import json
import torch.nn as nn
import os 
import pandas as pd
import joblib



app = Flask(__name__)
CORS(app)  # Enable CORS for your React app

class HeroRecommender(nn.Module):
    def __init__(self, num_heroes, num_roles, num_attributes, stat_dim=0,
                 hero_embedding_dim=128, role_embedding_dim=32, attr_embedding_dim=16):
        super(HeroRecommender, self).__init__()

        # Hero embeddings
        self.hero_embeddings = nn.Embedding(num_heroes, hero_embedding_dim, padding_idx=0)

        # Role embeddings
        self.role_embeddings = nn.Embedding(num_roles, role_embedding_dim)

        # Attribute embeddings
        self.attr_embeddings = nn.Embedding(num_attributes, attr_embedding_dim)

        # Stats processor (if stats are provided)
        self.use_stats = stat_dim > 0

        if self.use_stats:
            self.stats_processor = nn.Sequential(
                nn.Linear(stat_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.4),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.4),

                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.4),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.4),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
            )
            stats_output_dim = 128
        else:
            stats_output_dim = 0

        # Total embedding dimension per hero
        single_hero_dim = hero_embedding_dim + role_embedding_dim + attr_embedding_dim

        # Total input dimension
        total_input_dim = (single_hero_dim * 4) + stats_output_dim

        # Hidden dimensions
        hidden1 = total_input_dim*2
        hidden2 = total_input_dim*4
        hidden3 = total_input_dim*8
        hidden4 = total_input_dim*16

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(),
            nn.Dropout(0.55),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(),
            nn.Dropout(0.55),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.LeakyReLU(),
            nn.Dropout(0.55),

            nn.Linear(hidden3, hidden4),
            nn.BatchNorm1d(hidden4),
            nn.LeakyReLU(),
            nn.Dropout(0.55),

            nn.Linear(hidden4, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.LeakyReLU(),
            nn.Dropout(0.55),

            nn.Linear(hidden3, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(),
            nn.Dropout(0.55),

            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(),
            nn.Dropout(0.55),
        )

        # Output layer
        self.hero_decoder = nn.Linear(hidden1, num_heroes)

    def forward(self, hero_ids, attr_ids, role_ids, hero_stats=None):
        # Process each hero
        hero_embeds = []

        for i in range(4):  # Process 4 heroes
            # Get hero embedding
            hero_embed = self.hero_embeddings(hero_ids[:, i])

            # Get attribute embedding
            attr_embed = self.attr_embeddings(attr_ids[:, i])

            # Get role embedding
            role_embed = self.role_embeddings(role_ids[:, i])

            # Combine embeddings for this hero
            hero_combined = torch.cat([hero_embed, attr_embed, role_embed], dim=1)
            hero_embeds.append(hero_combined)

        # Process hero stats if given, and not NaN values
        if self.use_stats and hero_stats is not None:
            processed_stats = self.stats_processor(hero_stats)
            # Combine all hero embeddings and processed stats
            combined_input = torch.cat(hero_embeds + [processed_stats], dim=1)
        else:
            # When stats are enabled but not provided, create zeros tensor
            if self.use_stats:
                batch_size = hero_ids.shape[0]
                dummy_stats = torch.zeros(batch_size, 128, device=hero_ids.device)
                combined_input = torch.cat(hero_embeds + [dummy_stats], dim=1)
            else:
                combined_input = torch.cat(hero_embeds, dim=1)

        # Encode draft state
        encoded = self.encoder(combined_input)

        # Predict next hero
        hero_scores = self.hero_decoder(encoded)

        # Mask out already picked heroes
        hero_mask = torch.zeros_like(hero_scores, dtype=torch.bool)
        for i in range(hero_scores.size(0)):
            for j in range(4):
                hero_id = hero_ids[i, j].item()
                if hero_id > 0:  # Skip padding
                    hero_mask[i, hero_id] = True

            # Also mask padding index
            hero_mask[i, 0] = True

        hero_scores = hero_scores.masked_fill(hero_mask, -100)

        return hero_scores


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model3rdpick = pickle.load(open(os.path.join(BASE_DIR, 'draft_models', 'hero_recommender_3rd.pkl'), 'rb'))
model3rdpick.eval()
print("3rd hero prediction model loaded successfully")


model4thpick = pickle.load(open(os.path.join(BASE_DIR, 'draft_models', 'hero_recommender_4th.pkl'), 'rb'))
model4thpick.eval()
print("4th hero prediction model loaded successfully")


model5thpick = pickle.load(open(os.path.join(BASE_DIR, 'draft_models', 'hero_recommender_5th.pkl'), 'rb'))
model5thpick.eval()
print("5th hero prediction model loaded successfully")

winprobabilitymodel = joblib.load(open(os.path.join(BASE_DIR, 'draft_models', 'win_probability_model.pkl'), 'rb'))
print("Win probability model loaded successfully")
print(type(winprobabilitymodel))


    
    

@app.route('/')
def home():
    return "Model API functions!"

def format_hero_img(name):
    # Convert Dota localized name to image-safe string
    img_name = name.lower().replace(' ', '_').replace("'", '').replace('-', '_')
    
    # Hardcode known mismatches
    name_map = {
        'io': 'wisp',
        'storm spirit': 'storm_spirit',
        'queen of pain': 'queenofpain',
        'wraith king': 'skeleton_king',
        'magnus': 'magnataur',
        'outworld destroyer': 'obsidian_destroyer',
        'timbersaw': 'shredder',
        'centaur warrunner': 'centaur',
        'clockwerk': 'rattletrap',
        'windranger': 'windrunner',
        'zeus': 'zuus',
        'vengeful spirit': 'vengefulspirit'
    }

    final_name = name_map.get(name.lower(), img_name)
    return f"https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react/heroes/{final_name}.png"


# Load mappings and hero stats
meta=pd.read_csv(os.path.join(BASE_DIR, 'data', 'dota2_hero_stats.csv'))
print(meta.columns.tolist())


# Build hero metadata dictionary
hero_stat_columns = [
    'base_health', 'base_health_regen', 'base_mana', 'base_mana_regen',
    'base_armor', 'base_mr', 'base_attack_min', 'base_attack_max',
    'base_str', 'base_agi', 'base_int', 'str_gain', 'agi_gain', 'int_gain',
    'attack_range', 'attack_rate', 'base_attack_time',
    'attack_point', 'move_speed'
]
hero_metadata = {}

for index, row in meta.iterrows():
    hero_id = int(row['id'])

    # Build hero stat dictionary
    stats = {}
    for stat_name in hero_stat_columns:
        stats[stat_name] = row[stat_name]

    #Extract their data 
    hero_metadata[hero_id] = {
        'localized_name': row['localized_name'],  
        'primary_attr': row['primary_attr'],
        'roles': row['roles'].split(',') if isinstance(row['roles'], str) else [],
        'stats': stats,
        'img_url': format_hero_img(row['localized_name'])
    }

ATTRIBUTES = ['str', 'agi', 'int', 'all']
ROLES = ['Carry', 'Support', 'Nuker', 'Disabler', 'Jungler', 'Durable', 'Escape', 'Pusher', 'Initiator']

def encode_attr(attr):
    if attr == 'str':
        return [1, 0, 0, 0]
    elif attr == 'agi':
        return [0, 1, 0, 0]
    elif attr == 'int':
        return [0, 0, 1, 0]
    elif attr == 'all':
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0]  #Error Case
    

def encode_roles(role_list):
    role_vector = [0] * 9  # 9 roles

    if 'Carry' in role_list:
        role_vector[0] = 1
    if 'Support' in role_list:
        role_vector[1] = 1
    if 'Nuker' in role_list:
        role_vector[2] = 1
    if 'Disabler' in role_list:
        role_vector[3] = 1
    if 'Jungler' in role_list:
        role_vector[4] = 1
    if 'Durable' in role_list:
        role_vector[5] = 1
    if 'Escape' in role_list:
        role_vector[6] = 1
    if 'Pusher' in role_list:
        role_vector[7] = 1
    if 'Initiator' in role_list:
        role_vector[8] = 1

    return role_vector



def preprocess_input(picked_hero_ids):
    hero_ids = []
    attr_ids = []
    role_ids = []
    hero_stats = []

    for i, hero_id in enumerate(picked_hero_ids):
        if hero_id not in hero_metadata:
            continue

        hero = hero_metadata[hero_id]

        # Append ID for embedding
        hero_ids.append(hero_id)

        # Encode attribute and role
        attr_idx = ATTRIBUTES.index(hero["primary_attr"])
        attr_ids.append(attr_idx)

        primary_role = hero["roles"][0] if hero["roles"] else "Carry"
        role_idx = ROLES.index(primary_role)
        role_ids.append(role_idx)

        # Only take stats from the first 4 heroes
        if i < 4:
            stat_vec = [float(s) for s in hero["stats"].values()]
            hero_stats.extend(stat_vec)

    # Ensure exactly 4 heroes worth of stats (4×19 = 76)
    while len(hero_stats) < 76:
        hero_stats.extend([0.0] * (76 - len(hero_stats)))  # pad with zeros if needed

    hero_ids = torch.tensor(hero_ids).long().unsqueeze(0)         # e.g. [1, 6]
    attr_ids = torch.tensor(attr_ids).long().unsqueeze(0)         # e.g. [1, 6]
    role_ids = torch.tensor(role_ids).long().unsqueeze(0)         # e.g. [1, 6]
    hero_stats = torch.tensor(hero_stats).float().unsqueeze(0)    # shape: [1, 76]

    return hero_ids, attr_ids, role_ids, hero_stats









@app.route('/api/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    heroes = data.get('heroes', [])

    if len(heroes) == 4:
        # Pick the model you trained for 6 heroes
        model = model3rdpick

        # Preprocess
        hero_ids, attr_ids, role_ids, hero_stats = preprocess_input(heroes)

        model.eval()
        with torch.no_grad():
            output = model(hero_ids, attr_ids, role_ids, hero_stats)
            probs = torch.softmax(output, dim=1).squeeze().numpy()
            top10 = np.argsort(probs)[::-1][:10]

        suggestions = [
            {
                "hero_id": int(i),
                "name": hero_metadata[i]["localized_name"],
                "probability": float(probs[i]),
                "avatar": hero_metadata[i]["img_url"]
            }
            for i in top10
        ]

        return jsonify({"suggestions": suggestions})

    elif len(heroes) == 6:
        # Pick the model you trained for 6 heroes
        model = model4thpick

        # Preprocess
        hero_ids, attr_ids, role_ids, hero_stats = preprocess_input(heroes)

        model.eval()
        with torch.no_grad():
            output = model(hero_ids, attr_ids, role_ids, hero_stats)
            probs = torch.softmax(output, dim=1).squeeze().numpy()
            top10 = np.argsort(probs)[::-1][:10]

        suggestions = [
            {
                "hero_id": int(i),
                "name": hero_metadata[i]["localized_name"],
                "probability": float(probs[i]),
                "avatar": hero_metadata[i]["img_url"]
            }
            for i in top10
        ]

        return jsonify({"suggestions": suggestions})
    

    elif len(heroes)==8:
        # Pick the model you trained for 6 heroes
        model = model5thpick

        # Preprocess
        hero_ids, attr_ids, role_ids, hero_stats = preprocess_input(heroes)

        model.eval()
        with torch.no_grad():
            output = model(hero_ids, attr_ids, role_ids, hero_stats)
            probs = torch.softmax(output, dim=1).squeeze().numpy()
            top10 = np.argsort(probs)[::-1][:10]

        suggestions = [
            {
                "hero_id": int(i),
                "name": hero_metadata[i]["localized_name"],
                "probability": float(probs[i]),
                "avatar": hero_metadata[i]["img_url"]
            }
            for i in top10
        ]

        return jsonify({"suggestions": suggestions})
    
    else:
        print("ERROR")




  






def preprocess_win_input(hero_ids):
    # Make sure exactly 10 heroes are selected
    if len(hero_ids) != 10:
        raise ValueError("You must provide exactly 10 hero IDs.")

    # Load the correct feature column order (same as used during training)
    with open(os.path.join(BASE_DIR, 'data', 'hero_feature_columns.json')) as f:
        correct_order = json.load(f)

    # Start with all 0s
    one_hot_df = pd.DataFrame([[0] * len(correct_order)], columns=correct_order)

    # Set selected heroes to 1
    for id in hero_ids:
        col_name = f'hero_{id}'
        if col_name in one_hot_df.columns:
            one_hot_df.at[0, col_name] = 1
        else:
            print(f"Hero ID {id} not found in training columns")

    return one_hot_df



@app.route('/api/winprob', methods=['POST'])
def predict_win_probability():
    try:
        data = request.get_json()
        heroes = data.get("heroes", [])  # should be 10 total heroes

        if len(heroes) != 10:
            return jsonify({"error": "Expected 10 hero IDs (5 Radiant + 5 Dire)."}), 400

        # You'll replace this with your real preprocessing logic
        input_vector = preprocess_win_input(heroes)  

        # Make prediction
        win_prediction = winprobabilitymodel.predict(input_vector)[0]
        win_proba = winprobabilitymodel.predict_proba(input_vector)[0][1]  #Prob Radiant win

        return jsonify({
            "prediction": int(win_prediction),
            "radiant_win_probability": round(win_proba, 4)
        })

    except Exception as e:
        print("WinProb Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)