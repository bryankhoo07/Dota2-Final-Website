import { useState , useEffect} from 'react'
import dotaLogo from './assets/Dota_logo.svg.png'
import RadiantLogo from './assets/Cosmetic_icon_Radiant_Ancient.webp'
import DireLogo from './assets/Cosmetic_icon_Dire_Ancient.webp'
import Select from "./Select.jsx"
import './App.css'
import WinProbabilityChart from './Barchart.jsx'
import { heroesIdMapping } from './heroesIdMapping';







function App() {
  const [selectedHeroes, setSelectedHeroes] = useState(Array(10).fill(null));
  const [suggestedHeroes, setSuggestedHeroes] = useState([]);

  const [winProbability, setWinProbability] = useState(null); 
  
  const pickedHeroIds = selectedHeroes
  .filter(h => h !== null)
  .map(hero => hero.id);

  useEffect(() => {
    const picked = selectedHeroes.filter(h => h !== null);
    if (picked.length ==4||picked.length==6||picked.length==8) {
      getSuggestions();
    }
  }, [selectedHeroes]);

  const getSuggestions = async () => {
    const inputHeroes = selectedHeroes
      .filter(h => h !== null)
      .map(hero => heroesIdMapping[hero.name]);
  
    try {
      const response = await fetch('http://localhost:5000/api/suggest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ heroes: inputHeroes })
      });
  
      const data = await response.json();
      setSuggestedHeroes(data.suggestions); // Assuming backend returns `{ suggestions: [ { name, avatar }, ... ] }`
    } catch (error) {
      console.error("Error getting suggestions:", error);
    }
  };
  

  const getWinProbability = async () => {
    console.log("Button clicked, sending request...");
  
    const selectedHeroIds = selectedHeroes
      .filter(h => h !== null)
      .map(hero => heroesIdMapping[hero.name]);
  
    console.log("Selected hero IDs:", selectedHeroIds);
  
    try {
      const response = await fetch('http://localhost:5000/api/winprob', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ heroes: selectedHeroIds }),
      });
  
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
  
      const data = await response.json();
      console.log("Response from server:", data);
  
      if (data.radiant_win_probability !== undefined) {
        setWinProbability(data.radiant_win_probability);
      } else {
        console.warn("Win probability missing in response.");
      }
  
      if (data.suggestions) {
        setSuggestedHeroes(data.suggestions);
      }
  
    } catch (error) {
      console.error("Error getting win probability:", error);
    }
  };
  
  
  
  return (
    <>
      {/* Draft container with background image */}
      <div 
        className="bg-[url(./assets/dota-dota2-radiant-dire-logo.webp)] bg-left pt-[300px]  flex flex-col gap-[1px] w-full h-1/2 bg-cover bg-left"
      >
        {/* Navbar */}
        <div className="flex items-center text-white w-full absolute top-0 left-0 right-0 z-10">
          <a href="https://upload.wikimedia.org/wikipedia/commons/c/c2/Dota_logo.svg">
            <img 
              src={dotaLogo} 
              className="p-6 transition-all duration-300 hover:drop-shadow-[0_0_0.5em_rgba(100,108,255,0.67)]" 
              alt="Dota2 2 Logo"
            />
          </a>
          
          <div>
            <h1 className="pl-12">Dota 2 Draft Assistant</h1>  
            <h4>A draft assistant for Dota 2</h4>
          </div>
        </div>
        
        <h1>Draft</h1>
        
        {/* Team Headers */}
        <div className="absolute left-[10px]">
          <h1 className="text-green-500">Radiant</h1>
        </div>
        
        <div className="absolute right-0">
          <h1 className="text-red-500">Dire</h1>
        </div>
        
        {/* Hero Row 1 */}
        <div className="flex flex-row justify-between w-full h-full">
          <div className="block text-white h-40 w-40">
          <Select
            value={selectedHeroes[0]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[0] = val;
              setSelectedHeroes(updated);
            }}
            index={0}
          />
          </div>
          
          <div className="text-white">
            <h2>Hero1</h2>
          </div>
          
          <div className="block text-white h-40 w-40">
          <Select
            value={selectedHeroes[1]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[1] = val;
              setSelectedHeroes(updated);
            }}
            index={1}
          />
          </div>
        </div>
        
        {/* Hero Row 2 */}
        <div className="flex flex-row justify-between w-full h-full">
          <div className="block text-white h-40 w-40 ">
          <Select
            value={selectedHeroes[2]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[2] = val;
              setSelectedHeroes(updated);
            }}
            index={2}
          />
          </div>
          
          <div className="text-white">
            <h2>Hero 2</h2>
          </div>
          
          <div className="block text-white h-40 w-40">
          <Select
            value={selectedHeroes[3]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[3] = val;
              setSelectedHeroes(updated);
            }}
            index={3}
          />
          </div>
        </div>
        
        {/* Hero Row 3 */}
        <div className="flex flex-row justify-between w-full h-full">
          <div className="block text-white h-40 w-40 ">
          <Select
            value={selectedHeroes[4]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[4] = val;
              setSelectedHeroes(updated);
            }}
            index={4}
          />
          </div>
          
          <div className="text-white">
            <h2>Hero 3</h2>
          </div>
          
          <div className="block text-white h-40 w-40">
          <Select
            value={selectedHeroes[5]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[5] = val;
              setSelectedHeroes(updated);
            }}
            index={5}
          />
          </div>
        </div>
        
        {/* Hero Row 4 */}
        <div className="flex flex-row justify-between w-full h-full">
          <div className="block text-white h-40 w-40 ">
          <Select
            value={selectedHeroes[6]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[6] = val;
              setSelectedHeroes(updated);
            }}
            index={6}
          />
          </div>
          
          <div className="text-white">
            <h2>Hero 4</h2>
          </div>
          
          <div className="block text-white h-40 w-40 ">
          <Select
            value={selectedHeroes[7]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[7] = val;
              setSelectedHeroes(updated);
            }}
            index={7}
          />
          </div>
        </div>
        
        {/* Hero Row 5 */}
        <div className="flex flex-row justify-between w-full h-full">
          <div className="block text-white h-40 w-40 ">
          <Select
            value={selectedHeroes[8]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[8] = val;
              setSelectedHeroes(updated);
            }}
            index={8}
          />
          </div>
          
          <div className="text-white">
            <h2>Hero 5</h2>
          </div>
          
          <div className="block text-white h-40 w-40 ">
          <Select
            value={selectedHeroes[9]}
            onChange={(val) => {
              const updated = [...selectedHeroes];
              updated[9] = val;
              setSelectedHeroes(updated);
            }}
            index={9}
          />
          </div>
        </div>
      </div>

      {/* Suggestion and Chart Container */}
      <div className="flex w-full h-full justify-evenly  ">
        <div className="mt-auto flex-col text-center border-x-white">
          <h2 className="text-white relative">Suggestion</h2>
          <h2>This column suggestion to players what heroes to pick, Top 10 heroes</h2>
          <div className="max-h-[600px] overflow-y-auto grid grid-cols-2 gap-4">
            {suggestedHeroes.map((hero, index) => (
              <div key={index} className="text-white text-center">
                <img
                  src={hero.avatar}
                  alt={hero.name}
                  className="w-24 h-24 object-cover rounded border-2 border-green-500 mx-auto"
                />
                <h3 className="mt-2 text-lg font-semibold">{hero.name}</h3>
                <p className="text-sm text-gray-300">Score: {(hero.probability * 100).toFixed(2)}%</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex-col">
          <button
            onClick={getWinProbability}
            className="bg-blue-600 px-4 py-2 rounded text-white hover:bg-blue-700 mt-4"
          >
            Calculate Win Probability
          </button>

          <h2 className="align-top text-white">Win Probability Chart</h2>

          {winProbability !== null && (
            <>
              <WinProbabilityChart winProbability={winProbability} />
              <div className="text-lg font-semibold text-white mt-2 text-center">
                <p>
                  <span className="text-green-400">Radiant:</span>{" "}
                  <span className="text-yellow-300">{(winProbability * 100).toFixed(2)}%</span>
                </p>
                <p>
                  <span className="text-red-400">Dire:</span>{" "}
                  <span className="text-yellow-300">{(100 - winProbability * 100).toFixed(2)}%</span>
                </p>
              </div>
            </>
          )}
        </div> 
      </div>
    </>
  )
}

  
export default App