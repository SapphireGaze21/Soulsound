from flask import Flask, render_template, request, jsonify
from utils.emotion import EmotionAnalyzer
from utils.spotify import SpotifyRecommender

app = Flask(__name__)
emotion_analyzer = EmotionAnalyzer()
spotify_recommender = SpotifyRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    language = request.json.get('language', 'en')  # Default to English if not specified
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Analyze emotion
    emotion_result = emotion_analyzer.analyze_text(text)
    
    # Get song recommendations for each detected emotion
    emotion_recommendations = {}
    for emotion_data in emotion_result['emotions']:
        emotion = emotion_data['emotion']
        # Get 5 songs for each emotion
        songs = spotify_recommender.get_recommendations_for_emotion(
            emotion,
            language=language,
            limit=5
        )
        emotion_recommendations[emotion] = songs
        
        # If there are similar emotions, add their keywords to improve recommendations
        if 'similar_emotions' in emotion_data and emotion_data['similar_emotions']:
            # Get additional songs based on similar emotions
            for similar_emotion in emotion_data['similar_emotions']:
                similar_songs = spotify_recommender.get_recommendations_for_emotion(
                    similar_emotion,
                    language=language,
                    limit=2  # Get fewer songs for similar emotions
                )
                # Add to the main emotion's recommendations
                emotion_recommendations[emotion].extend(similar_songs)
            
            # Remove duplicates and limit to 5 songs
            seen = set()
            unique_songs = []
            for song in emotion_recommendations[emotion]:
                if song['name'] not in seen:
                    seen.add(song['name'])
                    unique_songs.append(song)
                if len(unique_songs) >= 5:
                    break
            emotion_recommendations[emotion] = unique_songs
    
    # Get general recommendations based on all keywords
    # Prioritize keywords from the primary emotion
    primary_emotion = emotion_result['primary_emotion']
    primary_keywords = emotion_analyzer.emotion_keywords.get(primary_emotion, [])
    
    # Combine primary keywords with all keywords, putting primary keywords first
    prioritized_keywords = primary_keywords + [k for k in emotion_result['keywords'] if k not in primary_keywords]
    
    general_recommendations = spotify_recommender.get_recommendations(
        prioritized_keywords,
        language=language,
        limit=5
    )
    
    return jsonify({
        'primary_emotion': emotion_result['primary_emotion'],
        'emotions': emotion_result['emotions'],
        'emotion_recommendations': emotion_recommendations,
        'general_recommendations': general_recommendations
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