import torch
import pickle
import torch.nn as nn
import json

# Define your model architecture - this is the same for all your models
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

# Define feature maps for all models - updated with correct dimensions from error message
feature_maps = {
    'num_heroes': 146,  # Updated from error message
    'num_roles': 9,     # Updated from error message
    'num_attributes': 4, # Using the original value, adjust if needed
    'stat_dim': 76      # Using the original value, adjust if needed
}

# List your models with their paths and descriptions
models_to_convert = [
    {
        'pth_file': 'Dota2 Draft Assistant/draft_models/hero_recommender_model_3rd.pth',  # Update with your actual file path
        'pkl_file': 'hero_recommender_3rd.pkl',
        'description': 'Model to predict 3rd hero',
        'feature_maps': feature_maps
    },
    {
        'pth_file': 'Dota2 Draft Assistant/draft_models/hero_recommender_model4th.pth',  # Update with your actual file path
        'pkl_file': 'hero_recommender_4th.pkl',
        'description': 'Model to predict 4th hero',
        'feature_maps': feature_maps
    },
    {
        'pth_file': 'Dota2 Draft Assistant/draft_models/hero_recommender_model5th.pth',  # Update with your actual file path
        'pkl_file': 'hero_recommender_5th.pkl',
        'description': 'Model to predict 5th hero',
        'feature_maps': feature_maps
    }
]

# Process each model
for model_info in models_to_convert:
    print(f"Converting {model_info['description']} model...")
    
    # Create a model instance with the same architecture
    model = HeroRecommender(
        num_heroes=model_info['feature_maps']['num_heroes'],
        num_roles=model_info['feature_maps']['num_roles'],
        num_attributes=model_info['feature_maps']['num_attributes'],
        stat_dim=model_info['feature_maps']['stat_dim']
    )
    
    try:
        # Load the weights from .pth file
        model.load_state_dict(torch.load(model_info['pth_file'], map_location=torch.device('cpu')))
        
        # Set model to evaluation mode
        model.eval()
        
        # Save as pickle file
        with open(model_info['pkl_file'], 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  - Model saved as {model_info['pkl_file']}")
    except Exception as e:
        print(f"  - Error converting model: {e}")

# Save feature maps for reference
with open('feature_maps.json', 'w') as f:
    json.dump(feature_maps, f, indent=2)

print("Conversion complete!")