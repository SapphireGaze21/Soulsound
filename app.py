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
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Analyze emotion
    emotion_result = emotion_analyzer.analyze_text(text)
    
    # Get song recommendations
    recommendations = spotify_recommender.get_recommendations(
        emotion_result['keywords']
    )
    
    return jsonify({
        'emotion': emotion_result['emotion'],
        'confidence': emotion_result['confidence'],
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True) 