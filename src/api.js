
const API_URL = 'http://localhost:5000/api';

export async function getRecommendations(allyHeroes, enemyHeroes) {
  try {
    const response = await fetch(`${API_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ allyHeroes, enemyHeroes })
    });
    
    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    const data = await response.json();
    return data.recommendations;
  } catch (error) {
    console.error('Error getting recommendations:', error);
    return [];
  }
}