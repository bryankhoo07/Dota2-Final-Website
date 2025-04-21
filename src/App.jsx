import { useState , useEffect} from 'react'
import dotaLogo from './assets/Dota_logo.svg.png'
import RadiantLogo from './assets/Cosmetic_icon_Radiant_Ancient.webp'
import DireLogo from './assets/Cosmetic_icon_Dire_Ancient.webp'
import Select from "./Select.jsx"
import './App.css'
import WinProbabilityChart from './Barchart.jsx'
import { heroesIdMapping } from './heroesIdMapping';







function App() {
  const [showInstructions, setShowInstructions] = useState(true);

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
      setSuggestedHeroes(data.suggestions); 
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
      const response = await fetch('http://localhost:5000/api/winprob',{
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
    {showInstructions && (
      
      <div className="fixed inset-0 bg-black bg-opacity-80 z-50 flex justify-center items-center">
        <div className="bg-white rounded-lg p-6 max-w-xl text-center shadow-lg">
        <h1 className="text-left text-black">Instructions</h1>
          <ul className="text-left list-disc list-inside mb-4 text-gray-800">
            <li>Pick heroes for Your Team (left) and Opponent's Team (right) using the dropdowns below.</li>
            <li>Once you have at least 2 Radiant and 2 Dire heroes selected, the assistant will suggest top picks at the bottom of the page</li>
            <li>The suggestor will constantly update for every pick from each team</li>
            <li>Click the blue button below to see the win probability after all 10 heroes are selected!</li>
          </ul>
          <button
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            onClick={() => setShowInstructions(false)}
          >
            Draft Begins!
          </button>
        </div>
      </div>
    )}
    
    
      {/* Draft container with Dota background image */}

      <div className="relative w-full min-h-screen mt-10">
      {/* Background image */}
      <div className="fixed inset-0 z-0 bg-color bg-left bg-no-repeat">
      <div className="w-full h-full 
                bg-[url('./assets/dota-dota2-radiant-dire-logo-laptopLarge.webp')] 
                sm:bg-[url('./assets/dota-dota2-radiant-dire-logo-Small.jpg')] 
                lg:bg-[url('./assets/dota-dota2-radiant-dire-logo-laptopLarge.webp')] 
                2xl:bg-[url('./assets/dota-dota2-radiant-dire-logo-4k.jpg')] 
                bg-cover bg-left" />

        <div className="absolute inset-0 bg-black opacity-30" />
      </div>
      {/* Foreground Content */}
        <div className="relative z-10 px-4 sm:px-8">
          {/* Navbar */}
          <div className="flex justify-center items-center text-white w-full top-10 left-0 right-0 z-10">
            <a href="Dota2 Draft Assistant/src/assets/Dota_logo.svg.png" className="flex items-center space-x-4">
              <img 
                src={dotaLogo} 
                className="h-12 w-auto transition-all duration-300 hover:drop-shadow-[0_0_0.5em_rgba(100,108,255,0.67)]" 
                alt="Dota2 Logo"
              />
              <h1 
                className="text-2xl sm:text-3xl font-bold" 
                style={{ fontFamily: 'Optiwtcgoudy' }}
              >
                Draft Assistant
              </h1>
            </a>
          </div>
      

        
        
        
        {/* Team Headers */}
        <div className="flex justify-between gap- px-4 sm:px-8 md:px-24 lg:px-60 mt-4">
          <h1 className="text-lime-500 font-bold text-xl sm:text-2xl text-center px-2" style={{ fontFamily: 'Optiwtcgoudy' }}>Radiant</h1>
          <h1 className="text-red-500 font-bold text-xl sm:text-2xl text-center px-2" style={{ fontFamily: 'Optiwtcgoudy' }}>Dire</h1>

        </div>


        <div className="max-w-screen-xl mx-auto px-4">
        {/* Hero Row 1 */}
        <div className="flex justify-between w-full h-full px-2 sm:px-4 md:px-16">
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
        <div className="flex justify-between w-full h-full px-2 sm:px-8 md:px-16">
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
        <div className="flex justify-between w-full h-full px-2 sm:px-8 md:px-16">
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
        <div className="flex justify-between w-full h-full px-2 sm:px-8 md:px-16">
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
        <div className="flex justify-between w-full h-full px-2 sm:px-8 md:px-16">
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
      <div className="flex flex-col md:flex-row w-full justify-evenly items-start mt-10 mb-10 px-4">
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
      </div>
      </div>
    
      
        
        
      
    </>
  )
}

  
export default App