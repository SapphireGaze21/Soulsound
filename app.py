from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from utils.emotion import EmotionAnalyzer
from utils.spotify import SpotifyRecommender
from utils.database import MongoDB
import random
import os
from functools import wraps
from bson.objectid import ObjectId
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Initialize services
emotion_analyzer = EmotionAnalyzer()
spotify_recommender = SpotifyRecommender()
db = MongoDB(app)

# Configure MongoDB
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/soulsound")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip authentication for API endpoints that use JSON
        if request.is_json:
            return f(*args, **kwargs)
        if 'username' not in session:
            flash('Please log in to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = db.verify_user(username, password)
        if user:
            session['username'] = username
            session['user_id'] = str(user['_id'])
            flash('Login successful!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_id = db.create_user(username, password)
        if user_id:
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Username already exists')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    flash('You have been logged out')
    return redirect(url_for('login'))

@app.route('/playlists', methods=['GET'])
@login_required
def get_playlists():
    user_id = session.get('user_id')
    if not user_id:
        print("Error: No user ID in session")  # Debug log
        return jsonify({'error': 'User not found', 'playlists': []}), 404
    
    print(f"Fetching playlists for user ID: {user_id}")  # Debug log
    playlists = db.get_user_playlists(user_id)
    print(f"Found {len(playlists)} playlists")  # Debug log
    
    return jsonify({
        'playlists': playlists or []  # Ensure we always return a list
    })

@app.route('/playlists', methods=['POST'])
@login_required
def create_playlist():
    try:
        print("Received create playlist request")  # Debug log
        data = request.get_json()
        print(f"Request data: {data}")  # Debug log
        
        name = data.get('name')
        user_id = session.get('user_id')
        
        print(f"Creating playlist - Name: {name}, User ID: {user_id}")  # Debug log
        
        if not name:
            print("Error: No playlist name provided")  # Debug log
            return jsonify({'error': 'Playlist name is required'}), 400
        
        if not user_id:
            print("Error: No user ID in session")  # Debug log
            return jsonify({'error': 'User not found'}), 404
        
        # Create the playlist
        playlist_id = db.create_playlist(name, user_id)
        print(f"Playlist created with ID: {playlist_id}")  # Debug log
        
        if playlist_id:
            return jsonify({
                'message': 'Playlist created successfully',
                'playlist_id': playlist_id
            })
        else:
            print("Error: Failed to create playlist in database")  # Debug log
            return jsonify({'error': 'Failed to create playlist'}), 500
            
    except Exception as e:
        print(f"Error in create_playlist route: {str(e)}")  # Debug log
        return jsonify({'error': 'An error occurred while creating playlist'}), 500

@app.route('/playlists/<playlist_id>', methods=['PUT'])
@login_required
def update_playlist(playlist_id):
    data = request.json
    new_name = data.get('name')
    
    if not new_name:
        return jsonify({'error': 'New playlist name is required'}), 400
    
    success = db.update_playlist_name(playlist_id, new_name)
    if success:
        return jsonify({'message': 'Playlist updated successfully'})
    return jsonify({'error': 'Playlist not found'}), 404

@app.route('/playlists/<playlist_id>', methods=['DELETE'])
@login_required
def delete_playlist(playlist_id):
    success = db.delete_playlist(playlist_id)
    if success:
        return jsonify({'message': 'Playlist deleted successfully'})
    return jsonify({'error': 'Playlist not found'}), 404

@app.route('/playlists/<playlist_id>/songs', methods=['POST'])
@login_required
def add_song_to_playlist(playlist_id):
    try:
        print(f"Adding song to playlist {playlist_id}")  # Debug log
        data = request.get_json()
        print(f"Request data: {data}")  # Debug log
        
        song = data.get('song')
        if not song:
            print("Error: No song data provided")  # Debug log
            return jsonify({'error': 'Song data is required'}), 400
        
        # Ensure song has required fields
        required_fields = ['name', 'artist', 'url']
        missing_fields = [field for field in required_fields if field not in song]
        if missing_fields:
            print(f"Error: Missing required fields: {missing_fields}")  # Debug log
            return jsonify({'error': f'Missing required song fields: {", ".join(missing_fields)}'}), 400
        
        # Add song to playlist
        success = db.add_song_to_playlist(playlist_id, song)
        print(f"Add song result: {success}")  # Debug log
        
        if success:
            return jsonify({'message': 'Song added to playlist successfully'})
        return jsonify({'error': 'Playlist not found'}), 404
        
    except Exception as e:
        print(f"Error in add_song_to_playlist: {str(e)}")  # Debug log
        return jsonify({'error': 'An error occurred while adding song to playlist'}), 500

@app.route('/playlists/<playlist_id>/songs/<song_id>', methods=['DELETE'])
@login_required
def remove_song_from_playlist(playlist_id, song_id):
    success = db.remove_song_from_playlist(playlist_id, song_id)
    if success:
        return jsonify({'message': 'Song removed from playlist successfully'})
    return jsonify({'error': 'Song or playlist not found'}), 404

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        text = request.json.get('text', '')
        language = request.json.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze emotion
        emotion_result = emotion_analyzer.analyze_text(text)
        
        return jsonify({
            'emotions': emotion_result['emotions']
        })
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

@app.route('/get-recommendations', methods=['POST'])
@login_required
def get_recommendations():
    try:
        selected_moods = request.json.get('selected_moods', [])
        language = request.json.get('language', 'en')
        
        if not selected_moods:
            return jsonify({'error': 'No moods selected'}), 400
        
        recommendations = []
        # Get 2 songs per mood to ensure a good mix
        songs_per_mood = 2
        
        for mood in selected_moods:
            try:
                songs = spotify_recommender.get_recommendations_for_emotion(
                    mood,
                    language=language,
                    limit=songs_per_mood
                )
                if songs:  # Only extend if we got songs back
                    recommendations.extend(songs)
            except Exception as e:
                print(f"Error getting recommendations for mood {mood}: {str(e)}")
                continue
        
        # Remove duplicates and limit to 5 songs
        seen = set()
        unique_recommendations = []
        for song in recommendations:
            if song['name'] not in seen:
                seen.add(song['name'])
                unique_recommendations.append(song)
            if len(unique_recommendations) >= 5:
                break
        
        if not unique_recommendations:
            return jsonify({'error': 'No recommendations found'}), 404
        
        return jsonify({
            'recommendations': unique_recommendations
        })
    except Exception as e:
        print(f"Error in get_recommendations route: {str(e)}")
        return jsonify({'error': 'An error occurred while getting recommendations'}), 500

@app.route('/shuffle', methods=['POST'])
@login_required
def shuffle():
    """
    Shuffle song recommendations based on selected moods and language.
    Improved version with better randomization and mood balancing.
    """
    try:
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
                    
                    if not songs:  # Skip if no songs returned
                        continue
                    
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
                    
                    if not songs:  # Skip if no songs returned
                        continue
                    
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
        
        if not recommendations:
            return jsonify({'error': 'No recommendations found'}), 404
        
        # Shuffle the final recommendations to ensure random order
        random.shuffle(recommendations)
        
        # Ensure we have exactly 5 songs
        recommendations = recommendations[:5]
        
        return jsonify({
            'recommendations': recommendations,
            'moods_used': selected_moods,
            'total_songs': len(recommendations)
        })
    except Exception as e:
        print(f"Error in shuffle route: {str(e)}")
        return jsonify({'error': 'An error occurred while shuffling recommendations'}), 500

@app.route('/test_emotion', methods=['GET'])
def test_emotion():
    """Test route to check if the emotion analyzer is working correctly."""
    try:
        test_text = "I am feeling very happy and excited today!"
        emotion_result = emotion_analyzer.analyze_text(test_text)
        return jsonify({
            'test_text': test_text,
            'emotion_result': emotion_result
        })
    except Exception as e:
        print(f"Error in test_emotion route: {str(e)}")
        return jsonify({'error': 'An error occurred while testing emotions'}), 500

@app.route('/playlists-page')
@login_required
def playlists_page():
    return render_template('playlists.html')

@app.route('/journals-page')
@login_required
def journals_page():
    return render_template('journals.html')

@app.route('/journals', methods=['GET'])
@login_required
def get_journals():
    user_id = session['user_id']
    journals = list(db.get_journals(user_id))
    # Convert ObjectId to string and format timestamp for JSON serialization
    for journal in journals:
        journal['_id'] = str(journal['_id'])
        # Convert timestamp to ISO format string
        if 'timestamp' in journal and journal['timestamp']:
            journal['timestamp'] = journal['timestamp'].isoformat()
    return jsonify({'journals': journals})

@app.route('/journals', methods=['POST'])
@login_required
def save_journal():
    user_id = session['user_id']
    content = request.json.get('content')
    if not content:
        return jsonify({'error': 'No content provided'}), 400
    
    journal_id = db.save_journal(user_id, content)
    return jsonify({'journal_id': str(journal_id)})

@app.route('/journals/<journal_id>', methods=['DELETE'])
@login_required
def delete_journal(journal_id):
    user_id = session['user_id']
    success = db.delete_journal(user_id, journal_id)
    if success:
        return jsonify({'message': 'Journal deleted successfully'})
    return jsonify({'error': 'Journal not found or unauthorized'}), 404

if __name__ == '__main__':
    # Use environment variable for port with a default of 5000
    port = int(os.environ.get("PORT", 5000))
    # In production, you might want to disable debug mode
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host='0.0.0.0', port=port, debug=debug) 