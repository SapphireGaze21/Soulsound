from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict

class EmotionAnalyzer:
    def __init__(self):
        # Initialize multiple models
        self.models = {
            # RoBERTa for detailed emotion analysis
            'roberta': pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            ),
            # BERT for general sentiment
            'bert': pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            ),
            # GoEmotions model for fine-grained emotions
            'goemotions': pipeline(
                "text-classification",
                model="monologg/bert-base-cased-goemotions-original",
                return_all_scores=True
            )
        }
        
        # Initialize NLTK sentiment analyzer
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion categories with expanded definitions
        self.emotion_categories = {
            'positive': ['joy', 'love', 'hope', 'gratitude', 'pride', 'contentment', 'excitement', 
                        'relief', 'curiosity', 'inspiration', 'optimism', 'trust', 'wonder',
                        'amusement', 'awe', 'desire', 'empathy', 'enthusiasm', 'gratitude',
                        'happiness', 'interest', 'joy', 'love', 'optimism', 'pride', 'relief',
                        'satisfaction', 'serenity', 'surprise', 'trust'],
            'negative': ['sadness', 'anger', 'fear', 'disappointment', 'confusion', 
                        'anxiety', 'boredom', 'embarrassment', 'envy', 'guilt', 'loneliness', 
                        'pessimism', 'regret', 'shame', 'annoyance', 'anxiety', 'boredom',
                        'confusion', 'disappointment', 'disgust', 'embarrassment', 'envy',
                        'fear', 'frustration', 'guilt', 'hate', 'irritation', 'jealousy',
                        'loneliness', 'nervousness', 'pessimism', 'regret', 'sadness',
                        'shame', 'stress', 'tension', 'worry'],
            'neutral': ['neutral', 'contemplation', 'nostalgia', 'determination', 'surprise',
                       'anticipation', 'attention', 'calmness', 'concentration', 'curiosity',
                       'determination', 'interest', 'meditation', 'neutral', 'observation',
                       'patience', 'reflection', 'serenity', 'thoughtfulness']
        }
        
        # Expanded emotion keywords for Spotify search
        self.emotion_keywords = {
            # Primary emotions
            'joy': ['happy', 'upbeat', 'cheerful', 'joyful', 'excited', 'elated', 'ecstatic', 'blissful'],
            'sadness': ['sad', 'melancholic', 'emotional', 'depressing', 'heartbroken', 'upset', 'gloomy', 'mournful'],
            'anger': ['angry', 'intense', 'powerful', 'furious', 'aggressive', 'enraged', 'irate', 'outraged'],
            'fear': ['anxious', 'nervous', 'scared', 'terrified', 'worried', 'apprehensive', 'frightened'],
            'love': ['romantic', 'love', 'passionate', 'affectionate', 'tender', 'care', 'devotion', 'adoration'],
            'surprise': ['energetic', 'exciting', 'dynamic', 'amazing', 'wonderful', 'wow', 'astonishing', 'stunning'],
            'neutral': ['balanced', 'moderate', 'steady', 'neutral', 'calm', 'peaceful', 'relaxing', 'serene'],
            
            # Additional emotions
            'hope': ['hopeful', 'inspiring', 'uplifting', 'motivational', 'encouraging', 'optimistic', 'positive'],
            'gratitude': ['grateful', 'thankful', 'appreciative', 'blessed', 'fortunate', 'thankful', 'appreciative'],
            'pride': ['proud', 'accomplished', 'successful', 'triumphant', 'victorious', 'achievement', 'success'],
            'nostalgia': ['nostalgic', 'memories', 'retro', 'vintage', 'classic', 'reminiscent', 'sentimental'],
            'contemplation': ['thoughtful', 'reflective', 'meditative', 'philosophical', 'deep', 'introspective', 'pensive'],
            'determination': ['determined', 'focused', 'driven', 'ambitious', 'resolute', 'persistent', 'tenacious'],
            'contentment': ['content', 'satisfied', 'fulfilled', 'at peace', 'serene', 'tranquil', 'satisfied'],
            'confusion': ['confused', 'uncertain', 'doubtful', 'questioning', 'searching', 'perplexed', 'bewildered'],
            'disappointment': ['disappointed', 'let down', 'frustrated', 'disheartened', 'discouraged', 'dismayed'],
            'excitement': ['excited', 'thrilled', 'enthusiastic', 'eager', 'energetic', 'animated', 'exhilarated'],
            
            # More specific emotions
            'awe': ['awe-inspiring', 'majestic', 'grand', 'magnificent', 'sublime', 'wonder', 'amazement'],
            'desire': ['desirous', 'longing', 'yearning', 'craving', 'passionate', 'eager', 'ardent'],
            'empathy': ['empathetic', 'compassionate', 'understanding', 'sympathetic', 'caring', 'kind'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'exasperated', 'aggravated', 'bothered'],
            'guilt': ['guilty', 'remorseful', 'regretful', 'apologetic', 'contrite', 'repentant'],
            'jealousy': ['jealous', 'envious', 'covetous', 'resentful', 'begrudging', 'green-eyed'],
            'loneliness': ['lonely', 'isolated', 'alone', 'solitary', 'forsaken', 'abandoned'],
            'optimism': ['optimistic', 'positive', 'hopeful', 'confident', 'upbeat', 'cheerful'],
            'pessimism': ['pessimistic', 'negative', 'cynical', 'doubtful', 'skeptical', 'gloomy'],
            'regret': ['regretful', 'remorseful', 'sorry', 'apologetic', 'contrite', 'repentant'],
            'shame': ['ashamed', 'embarrassed', 'humiliated', 'disgraced', 'mortified', 'chagrined'],
            'trust': ['trusting', 'faithful', 'loyal', 'reliable', 'dependable', 'confident'],
            'wonder': ['wondering', 'amazed', 'astonished', 'awestruck', 'marveling', 'fascinated']
        }

    def analyze_text(self, text):
        """
        Analyze the emotional content of the text using multiple models and approaches.
        """
        # Get predictions from all models
        results = {}
        for model_name, model in self.models.items():
            try:
                results[model_name] = model(text)[0]
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                continue
        
        # Get NLTK sentiment scores
        nltk_scores = self.sia.polarity_scores(text)
        
        # Combine results from different models
        combined_scores = defaultdict(float)
        total_weight = 0
        
        # Weight for each model
        weights = {
            'roberta': 0.4,  # Primary emotion model
            'bert': 0.3,     # General sentiment
            'goemotions': 0.3, # Fine-grained emotions
            'nltk': 0.2      # Lexicon-based approach
        }
        
        # Process each model's results
        for model_name, model_results in results.items():
            if model_name == 'roberta' or model_name == 'goemotions':
                for result in model_results:
                    emotion = result['label'].lower()
                    score = result['score']
                    combined_scores[emotion] += score * weights[model_name]
            elif model_name == 'bert':
                # Convert sentiment to emotion scores
                sentiment = model_results[0]['label']
                score = model_results[0]['score']
                if sentiment == 'POSITIVE':
                    combined_scores['joy'] += score * weights[model_name]
                elif sentiment == 'NEGATIVE':
                    combined_scores['sadness'] += score * weights[model_name]
                else:
                    combined_scores['neutral'] += score * weights[model_name]
        
        # Add NLTK scores
        for emotion, score in nltk_scores.items():
            if emotion == 'pos':
                combined_scores['joy'] += score * weights['nltk']
            elif emotion == 'neg':
                combined_scores['sadness'] += score * weights['nltk']
            elif emotion == 'neu':
                combined_scores['neutral'] += score * weights['nltk']
        
        # Normalize scores
        total_score = sum(combined_scores.values())
        if total_score > 0:
            combined_scores = {k: v/total_score for k, v in combined_scores.items()}
        
        # Get top emotions
        sorted_emotions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare final result
        detected_emotions = []
        for emotion, confidence in sorted_emotions:
            if confidence > 0.1:  # Only include emotions with significant confidence
                detected_emotions.append({
                    'emotion': emotion,
                    'confidence': confidence
                })
        
        # Get keywords for all detected emotions
        all_keywords = []
        for emotion_data in detected_emotions:
            emotion = emotion_data['emotion']
            if emotion in self.emotion_keywords:
                all_keywords.extend(self.emotion_keywords[emotion])
        
        # Remove duplicates
        all_keywords = list(set(all_keywords))
        
        # Determine the primary emotion
        primary_emotion = detected_emotions[0]['emotion'] if detected_emotions else 'neutral'
        
        result = {
            'primary_emotion': primary_emotion,
            'emotions': detected_emotions,
            'keywords': all_keywords,
            'raw_scores': {
                'nltk': nltk_scores,
                'models': {k: v for k, v in results.items()}
            }
        }
        
        return result 