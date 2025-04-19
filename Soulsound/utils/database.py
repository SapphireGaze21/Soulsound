from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import hashlib
import os

class MongoDB:
    def __init__(self, app):
        # MongoDB configuration
        app.config["MONGO_URI"] = "mongodb://localhost:27017/soulsound"
        print(f"Connecting to MongoDB at: {app.config['MONGO_URI']}")  # Debug log
        
        try:
            self.mongo = PyMongo(app)
            self.db = self.mongo.db
            self.users = self.db.users
            print("MongoDB connection successful")  # Debug log
            print(f"Database name: {self.db.name}")  # Debug log
        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")  # Debug log
            raise

    def create_playlist(self, name, user_id=None):
        """Create a new playlist"""
        try:
            print(f"Creating playlist in MongoDB - Name: {name}, User ID: {user_id}")  # Debug log
            
            # Validate inputs
            if not name:
                print("Error: No playlist name provided")
                return None
            
            if not user_id:
                print("Error: No user ID provided")
                return None
            
            # Check if user exists
            user = self.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                print(f"Error: User with ID {user_id} not found")
                return None
            
            playlist = {
                'name': name,
                'user_id': user_id,
                'songs': [],
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            print(f"Playlist data: {playlist}")  # Debug log
            
            result = self.db.playlists.insert_one(playlist)
            print(f"MongoDB insert result: {result.inserted_id}")  # Debug log
            
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error in create_playlist: {str(e)}")  # Debug log
            return None

    def add_song_to_playlist(self, playlist_id, song):
        """Add a song to a playlist"""
        song['added_at'] = datetime.utcnow()
        result = self.db.playlists.update_one(
            {'_id': ObjectId(playlist_id)},
            {
                '$push': {'songs': song},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        return result.modified_count > 0

    def remove_song_from_playlist(self, playlist_id, song_id):
        """Remove a song from a playlist"""
        result = self.db.playlists.update_one(
            {'_id': ObjectId(playlist_id)},
            {
                '$pull': {'songs': {'id': song_id}},
                '$set': {'updated_at': datetime.utcnow()}
            }
        )
        return result.modified_count > 0

    def get_playlist(self, playlist_id):
        """Get a playlist by ID"""
        playlist = self.db.playlists.find_one({'_id': ObjectId(playlist_id)})
        if playlist:
            playlist['_id'] = str(playlist['_id'])
        return playlist

    def get_user_playlists(self, user_id):
        """Get all playlists for a user"""
        playlists = list(self.db.playlists.find({'user_id': user_id}))
        for playlist in playlists:
            playlist['_id'] = str(playlist['_id'])
        return playlists

    def delete_playlist(self, playlist_id):
        """Delete a playlist"""
        result = self.db.playlists.delete_one({'_id': ObjectId(playlist_id)})
        return result.deleted_count > 0

    def update_playlist_name(self, playlist_id, new_name):
        """Update playlist name"""
        result = self.db.playlists.update_one(
            {'_id': ObjectId(playlist_id)},
            {
                '$set': {
                    'name': new_name,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    # User-related functions
    def create_user(self, username, password):
        """Create a new user with hashed password"""
        # Check if username already exists
        if self.users.find_one({'username': username}):
            return None
        
        # Hash the password
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        
        # Store user with hashed password
        user = {
            'username': username,
            'password': key,
            'salt': salt
        }
        
        result = self.users.insert_one(user)
        return str(result.inserted_id)

    def verify_user(self, username, password):
        """Verify user credentials"""
        user = self.users.find_one({'username': username})
        if not user:
            return None
        
        # Check if user has the new password format with salt
        if 'salt' in user and 'password' in user:
            # Verify password with salt
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                user['salt'],
                100000
            )
            if key == user['password']:
                return user
        else:
            # For users with old password format, update to new format
            # Generate new salt and hash
            salt = os.urandom(32)
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            
            # Update user with new password format
            self.users.update_one(
                {'_id': user['_id']},
                {
                    '$set': {
                        'password': key,
                        'salt': salt
                    }
                }
            )
            return user
            
        return None

    def get_user_by_id(self, user_id):
        """
        Get user by ID.
        Returns the user document if found, None otherwise.
        """
        try:
            user = self.db.users.find_one({"_id": ObjectId(user_id)})
            if user:
                # Don't return the password hash
                user.pop("password", None)
                return user
            return None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None 