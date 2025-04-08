import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

class SpotifyRecommender:
    def __init__(self):
        load_dotenv()
        
        # Initialize Spotify client
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
            )
        )

    def get_recommendations(self, keywords, limit=3):
        """
        Get song recommendations based on emotion keywords.
        Returns a list of songs with their Spotify links.
        """
        recommendations = []
        
        for keyword in keywords:
            # Search for tracks using the keyword
            results = self.sp.search(
                q=keyword,
                type='track',
                limit=limit
            )
            
            # Extract relevant information from each track
            for track in results['tracks']['items']:
                recommendations.append({
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                })
            
            # If we have enough recommendations, break
            if len(recommendations) >= limit:
                break
        
        # Return only the requested number of recommendations
        return recommendations[:limit] 