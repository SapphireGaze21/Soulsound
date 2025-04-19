from flask import Flask, render_template, request, jsonify
from utils.emotion import EmotionAnalyzer
from utils.spotify import SpotifyRecommender
import random

app = Flask(__name__)
emotion_analyzer = EmotionAnalyzer()
spotify_recommender = SpotifyRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    language = request.json.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Analyze emotion
    emotion_result = emotion_analyzer.analyze_text(text)
    
    return jsonify({
        'emotions': emotion_result['emotions']
    })

@app.route('/get-recommendations', methods=['POST'])
def get_recommendations():
    selected_moods = request.json.get('selected_moods', [])
    language = request.json.get('language', 'en')
    
    if not selected_moods:
        return jsonify({'error': 'No moods selected'}), 400
    
    recommendations = []
    # Get 2 songs per mood to ensure a good mix
    songs_per_mood = 2
    
    for mood in selected_moods:
        songs = spotify_recommender.get_recommendations_for_emotion(
            mood,
            language=language,
            limit=songs_per_mood
        )
        recommendations.extend(songs)
    
    # Remove duplicates and limit to 5 songs
    seen = set()
    unique_recommendations = []
    for song in recommendations:
        if song['name'] not in seen:
            seen.add(song['name'])
            unique_recommendations.append(song)
        if len(unique_recommendations) >= 5:
            break
    
    return jsonify({
        'recommendations': unique_recommendations
    })

@app.route('/shuffle', methods=['POST'])
def shuffle():
    """
    Shuffle song recommendations based on selected moods and language.
    Improved version with better randomization and mood balancing.
    """
    data = request.json
    language = data.get('language', 'en')
    selected_moods = data.get('selected_moods', [])
    
    if not selected_moods:
        return jsonify({'error': 'No moods selected'}), 400
    
    # Calculate number of songs per mood based on total moods
    total_moods = len(selected_moods)
    songs_per_mood = max(2, min(3, 5 // total_moods))  # Adjust songs per mood based on total moods
    
    # Generate multiple random offsets for better variety
    offsets = [random.randint(0, 100) for _ in range(3)]  # Generate 3 different offsets
    
    recommendations = []
    seen_songs = set()
    
    # Try each offset to get different sets of songs
    for offset in offsets:
        for mood in selected_moods:
            try:
                songs = spotify_recommender.get_recommendations_for_emotion(
                    mood,
                    language=language,
                    limit=songs_per_mood,
                    offset=offset
                )
                
                # Add only new songs to recommendations
                for song in songs:
                    if song['name'] not in seen_songs:
                        seen_songs.add(song['name'])
                        recommendations.append(song)
                        
                        # If we have enough songs, break early
                        if len(recommendations) >= 5:
                            break
                            
                if len(recommendations) >= 5:
                    break
                    
            except Exception as e:
                print(f"Error getting recommendations for mood {mood}: {str(e)}")
                continue
                
        if len(recommendations) >= 5:
            break
    
    # If we still don't have enough songs, try one more time with default offset
    if len(recommendations) < 5:
        remaining_slots = 5 - len(recommendations)
        for mood in selected_moods:
            try:
                songs = spotify_recommender.get_recommendations_for_emotion(
                    mood,
                    language=language,
                    limit=remaining_slots
                )
                
                for song in songs:
                    if song['name'] not in seen_songs:
                        seen_songs.add(song['name'])
                        recommendations.append(song)
                        
                        if len(recommendations) >= 5:
                            break
                            
                if len(recommendations) >= 5:
                    break
                    
            except Exception as e:
                print(f"Error getting final recommendations for mood {mood}: {str(e)}")
                continue
    
    # Shuffle the final recommendations to ensure random order
    random.shuffle(recommendations)
    
    # Ensure we have exactly 5 songs
    recommendations = recommendations[:5]
    
    return jsonify({
        'recommendations': recommendations,
        'moods_used': selected_moods,
        'total_songs': len(recommendations)
    })

@app.route('/test_emotion', methods=['GET'])
def test_emotion():
    """Test route to check if the emotion analyzer is working correctly."""
    test_text = "I am feeling very happy and excited today!"
    emotion_result = emotion_analyzer.analyze_text(test_text)
    return jsonify({
        'test_text': test_text,
        'emotion_result': emotion_result
    })

if __name__ == '__main__':
    app.run(debug=True) 