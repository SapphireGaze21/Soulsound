from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from utils.emotion import EmotionAnalyzer
from utils.spotify import SpotifyRecommender
from utils.mongodb import MongoDB
import random
from functools import wraps
import os

app = Flask(__name__)
# Add secret key for session management
app.secret_key = os.urandom(24)
emotion_analyzer = EmotionAnalyzer()
spotify_recommender = SpotifyRecommender()
mongodb = MongoDB(app)

# Add this function to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Add these routes for authentication
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = mongodb.verify_user(username, password)
        if user:
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    # Get success message from registration if any
    message = request.args.get('message')
    return render_template('login.html', message=message)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters long')
        
        user_id = mongodb.create_user(username, password)
        if user_id:
            # Instead of logging in, redirect to login page with success message
            return redirect(url_for('login', message='Registration successful! Please login with your credentials.'))
        else:
            return render_template('register.html', error='Username already exists')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    # Clear Flask session
    session.clear()
    # Return a response that will also clear browser session storage
    response = redirect(url_for('login'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Update the index route to require login
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# Update the playlists route to require login
@app.route('/playlists')
@login_required
def playlists_page():
    return render_template('playlists.html')

@app.route('/analyze', methods=['POST'])
@login_required
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
        # Get 3 songs for each emotion
        songs = spotify_recommender.get_recommendations_for_emotion(
            emotion, 
            language=language,
            limit=3
        )
        emotion_recommendations[emotion] = songs
        
        # If there are similar emotions, add their keywords to improve recommendations
        if 'similar_emotions' in emotion_data and emotion_data['similar_emotions']:
            # Get additional songs based on similar emotions
            for similar_emotion in emotion_data['similar_emotions']:
                similar_songs = spotify_recommender.get_recommendations_for_emotion(
                    similar_emotion,
                    language=language,
                    limit=1  # Get fewer songs for similar emotions
                )
                # Add to the main emotion's recommendations
                emotion_recommendations[emotion].extend(similar_songs)
            
            # Remove duplicates and limit to 3 songs
            seen = set()
            unique_songs = []
            for song in emotion_recommendations[emotion]:
                if song['name'] not in seen:
                    seen.add(song['name'])
                    unique_songs.append(song)
                if len(unique_songs) >= 3:
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
        limit=3
    )
    
    return jsonify({
        'primary_emotion': emotion_result['primary_emotion'],
        'emotions': emotion_result['emotions'],
        'emotion_recommendations': emotion_recommendations,
        'general_recommendations': general_recommendations,
        'keywords': prioritized_keywords
    })

@app.route('/shuffle', methods=['POST'])
def shuffle():
    """
    Shuffle song recommendations for a specific emotion or get new general recommendations.
    """
    data = request.json
    emotion = data.get('emotion', '')
    language = data.get('language', 'en')
    is_general = data.get('is_general', False)
    
    if not emotion and not is_general:
        return jsonify({'error': 'No emotion provided'}), 400
    
    # Generate a random offset to get different songs
    offset = random.randint(5, 20)
    
    if is_general:
        # Get new general recommendations
        keywords = data.get('keywords', [])
        if not keywords:
            return jsonify({'error': 'No keywords provided for general recommendations'}), 400
        
        recommendations = spotify_recommender.get_recommendations(
            keywords,
            language=language,
            limit=3,
            offset=offset
        )
        
        return jsonify({
            'recommendations': recommendations
        })
    else:
        # Get new recommendations for the specific emotion
        recommendations = spotify_recommender.get_recommendations_for_emotion(
            emotion,
            language=language,
            limit=3,
            offset=offset
        )
        
        return jsonify({
            'recommendations': recommendations
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

@app.route('/playlists', methods=['GET'])
@login_required
def get_playlists():
    user_id = session.get('user_id')
    playlists = mongodb.get_user_playlists(user_id)
    return jsonify(playlists)

@app.route('/playlists', methods=['POST'])
@login_required
def create_playlist():
    data = request.json
    name = data.get('name')
    user_id = session.get('user_id')
    
    if not name:
        return jsonify({'error': 'Playlist name is required'}), 400
    
    playlist_id = mongodb.create_playlist(name, user_id)
    if playlist_id:
        return jsonify({'playlist_id': playlist_id}), 201
    else:
        return jsonify({'error': 'Failed to create playlist'}), 500

@app.route('/playlists/<playlist_id>/songs', methods=['POST'])
def add_song_to_playlist(playlist_id):
    """Add a song to a playlist"""
    song = request.json
    if not song:
        return jsonify({'error': 'Song data is required'}), 400
    
    success = mongodb.add_song_to_playlist(playlist_id, song)
    if success:
        return jsonify({'message': 'Song added to playlist'})
    return jsonify({'error': 'Failed to add song to playlist'}), 400

@app.route('/playlists/<playlist_id>/songs/<song_id>', methods=['DELETE'])
def remove_song_from_playlist(playlist_id, song_id):
    """Remove a song from a playlist"""
    success = mongodb.remove_song_from_playlist(playlist_id, song_id)
    if success:
        return jsonify({'message': 'Song removed from playlist'})
    return jsonify({'error': 'Failed to remove song from playlist'}), 400

@app.route('/playlists/<playlist_id>', methods=['GET'])
def get_playlist(playlist_id):
    """Get a specific playlist"""
    playlist = mongodb.get_playlist(playlist_id)
    if playlist:
        return jsonify(playlist)
    return jsonify({'error': 'Playlist not found'}), 404

@app.route('/playlists/<playlist_id>', methods=['DELETE'])
def delete_playlist(playlist_id):
    """Delete a playlist"""
    success = mongodb.delete_playlist(playlist_id)
    if success:
        return jsonify({'message': 'Playlist deleted'})
    return jsonify({'error': 'Failed to delete playlist'}), 400

@app.route('/playlists/<playlist_id>', methods=['PUT'])
def update_playlist(playlist_id):
    """Update playlist name"""
    data = request.json
    new_name = data.get('name')
    if not new_name:
        return jsonify({'error': 'New name is required'}), 400
    
    success = mongodb.update_playlist_name(playlist_id, new_name)
    if success:
        return jsonify({'message': 'Playlist updated'})
    return jsonify({'error': 'Failed to update playlist'}), 400

if __name__ == '__main__':
    app.run(debug=True) 