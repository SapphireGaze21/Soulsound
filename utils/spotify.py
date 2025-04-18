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
        
        # Language-specific search parameters
        self.language_markets = {
            # International Languages
            'en': 'US',  # English - United States
            'es': 'ES',  # Spanish - Spain
            'fr': 'FR',  # French - France
            'de': 'DE',  # German - Germany
            'it': 'IT',  # Italian - Italy
            'pt': 'BR',  # Portuguese - Brazil
            'ru': 'RU',  # Russian - Russia
            'ja': 'JP',  # Japanese - Japan
            'ko': 'KR',  # Korean - South Korea
            'zh': 'TW',  # Chinese - Taiwan
            
            # Indian Languages
            'hi': 'IN',  # Hindi - India
            'bn': 'IN',  # Bengali - India
            'ta': 'IN',  # Tamil - India
            'te': 'IN',  # Telugu - India
            'kn': 'IN',  # Kannada - India
            'ml': 'IN',  # Malayalam - India
            'gu': 'IN',  # Gujarati - India
            'pa': 'IN',  # Punjabi - India
            'mr': 'IN',  # Marathi - India
            'ur': 'IN'   # Urdu - India
        }
        
        # Language-specific search terms for all languages
        self.language_terms = {
            # International Languages
            'en': ['english'],
            'es': ['spanish', 'español'],
            'fr': ['french', 'français'],
            'de': ['german', 'deutsch'],
            'it': ['italian', 'italiano'],
            'pt': ['portuguese', 'português'],
            'ru': ['russian', 'русский'],
            'ja': ['japanese', '日本語'],
            'ko': ['korean', '한국어'],
            'zh': ['chinese', '中文'],
            
            # Indian Languages
            'hi': ['hindi', 'bollywood'],
            'bn': ['bengali', 'bangla'],
            'ta': ['tamil', 'kollywood'],
            'te': ['telugu', 'tollywood'],
            'kn': ['kannada', 'sandalwood'],
            'ml': ['malayalam', 'mollywood'],
            'gu': ['gujarati'],
            'pa': ['punjabi', 'bhangra'],
            'mr': ['marathi'],
            'ur': ['urdu', 'ghazal']
        }
        
        # Language-specific genre tags
        self.language_genres = {
            # International Languages
            'en': ['pop', 'rock'],
            'es': ['latin', 'flamenco', 'reggaeton'],
            'fr': ['chanson', 'french pop'],
            'de': ['schlager', 'volksmusik'],
            'it': ['italian pop', 'opera'],
            'pt': ['samba', 'bossa nova', 'mpb'],
            'ru': ['russian pop', 'folk'],
            'ja': ['j-pop', 'anime'],
            'ko': ['k-pop', 'korean pop'],
            'zh': ['mandopop', 'c-pop'],
            
            # Indian Languages
            'hi': ['bollywood', 'hindi pop'],
            'bn': ['bengali pop', 'rabindra sangeet'],
            'ta': ['tamil pop', 'kollywood'],
            'te': ['telugu pop', 'tollywood'],
            'kn': ['kannada pop', 'carnatic'],
            'ml': ['malayalam pop', 'mollywood'],
            'gu': ['gujarati pop', 'garba'],
            'pa': ['punjabi pop', 'bhangra'],
            'mr': ['marathi pop', 'lavani'],
            'ur': ['urdu pop', 'ghazal', 'qawwali']
        }

    def get_recommendations_for_emotion(self, emotion, language='en', limit=3, offset=0):
        """
        Get song recommendations for a specific emotion and language.
        Returns a mix of popular and underrated songs.
        """
        recommendations = []
        market = self.language_markets.get(language, 'US')
        
        # Get language-specific terms and genres
        language_terms = self.language_terms.get(language, [])
        language_genres = self.language_genres.get(language, [])
        
        # First try with language-specific genre for popular songs
        for genre in language_genres:
            # Create a search query that combines emotion keyword, language term, and genre
            search_query = f"{emotion} genre:{genre}"
            
            # Search for popular tracks
            results = self.sp.search(
                q=search_query,
                type='track',
                limit=limit,
                offset=offset,
                market=market
            )
            
            # Sort tracks by popularity
            tracks = sorted(results['tracks']['items'], key=lambda x: x['popularity'], reverse=True)
            
            # Add top popular tracks
            for track in tracks[:limit//2]:  # Get half of the limit as popular songs
                if track['popularity'] > 50:  # Only include relatively popular songs
                    recommendations.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url'],
                        'popularity': track['popularity']
                    })
            
            # Add underrated tracks
            for track in tracks[limit//2:]:  # Get the other half as underrated songs
                if track['popularity'] < 50:  # Only include less popular songs
                    recommendations.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url'],
                        'popularity': track['popularity']
                    })
            
            # If we have enough recommendations, break
            if len(recommendations) >= limit:
                break
        
        # If we still need more recommendations, try with language terms
        if len(recommendations) < limit:
            for lang_term in language_terms:
                search_query = f"{emotion} {lang_term}"
                
                results = self.sp.search(
                    q=search_query,
                    type='track',
                    limit=limit,
                    offset=offset,
                    market=market
                )
                
                # Sort tracks by popularity
                tracks = sorted(results['tracks']['items'], key=lambda x: x['popularity'], reverse=True)
                
                # Add popular tracks
                for track in tracks[:limit//2]:
                    if track['popularity'] > 50 and len(recommendations) < limit:
                        recommendations.append({
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'url': track['external_urls']['spotify'],
                            'preview_url': track['preview_url'],
                            'popularity': track['popularity']
                        })
                
                # Add underrated tracks
                for track in tracks[limit//2:]:
                    if track['popularity'] < 50 and len(recommendations) < limit:
                        recommendations.append({
                            'name': track['name'],
                            'artist': track['artists'][0]['name'],
                            'url': track['external_urls']['spotify'],
                            'preview_url': track['preview_url'],
                            'popularity': track['popularity']
                        })
                
                if len(recommendations) >= limit:
                    break
        
        # If we still need more recommendations, try with just the keyword
        if len(recommendations) < limit:
            results = self.sp.search(
                q=emotion,
                type='track',
                limit=limit,
                offset=offset,
                market=market
            )
            
            # Sort tracks by popularity
            tracks = sorted(results['tracks']['items'], key=lambda x: x['popularity'], reverse=True)
            
            # Add remaining tracks
            for track in tracks:
                if len(recommendations) < limit:
                    recommendations.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url'],
                        'popularity': track['popularity']
                    })
        
        # Sort recommendations to mix popular and underrated songs
        recommendations.sort(key=lambda x: x['popularity'], reverse=True)
        mixed_recommendations = []
        
        # Interleave popular and underrated songs
        popular = [r for r in recommendations if r['popularity'] > 50]
        underrated = [r for r in recommendations if r['popularity'] <= 50]
        
        for i in range(max(len(popular), len(underrated))):
            if i < len(popular):
                mixed_recommendations.append(popular[i])
            if i < len(underrated):
                mixed_recommendations.append(underrated[i])
        
        # Return only the requested number of recommendations
        return mixed_recommendations[:limit]

    def get_recommendations(self, keywords, language='en', limit=3, offset=0):
        """
        Get song recommendations based on emotion keywords and language.
        Returns a list of songs with their Spotify links.
        
        Args:
            keywords: List of keywords to search for
            language: The language code (default: 'en')
            limit: Maximum number of recommendations to return (default: 3)
            offset: Offset for pagination to get different results (default: 0)
        """
        recommendations = []
        market = self.language_markets.get(language, 'US')  # Default to US if language not found
        
        # Get language-specific terms and genres
        language_terms = self.language_terms.get(language, [])
        language_genres = self.language_genres.get(language, [])
        
        # Create a combined search query with language and emotion
        for keyword in keywords:
            # First try with language-specific genre
            for genre in language_genres:
                # Create a search query that combines keyword, language term, and genre
                search_query = f"{keyword} genre:{genre}"
                
                # Search for tracks using the combined query and language-specific market
                results = self.sp.search(
                    q=search_query,
                    type='track',
                    limit=limit,
                    offset=offset,  # Add offset for pagination
                    market=market
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
            
            # If we still need more recommendations, try with language terms
            if len(recommendations) < limit:
                for lang_term in language_terms:
                    # Create a search query that combines keyword and language term
                    search_query = f"{keyword} {lang_term}"
                    
                    # Search for tracks using the combined query and language-specific market
                    results = self.sp.search(
                        q=search_query,
                        type='track',
                        limit=limit,
                        offset=offset,  # Add offset for pagination
                        market=market
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
            
            # If we still need more recommendations, try with just the keyword in the selected market
            if len(recommendations) < limit:
                results = self.sp.search(
                    q=keyword,
                    type='track',
                    limit=limit,
                    offset=offset,  # Add offset for pagination
                    market=market
                )
                
                # Extract relevant information from each track
                for track in results['tracks']['items']:
                    recommendations.append({
                        'name': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'preview_url': track['preview_url']
                    })
        
        # Return only the requested number of recommendations
        return recommendations[:limit] 