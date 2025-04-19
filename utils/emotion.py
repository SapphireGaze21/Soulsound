from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict
import re

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
        
        # Activity-based emotion mapping
        self.activity_emotions = {
            # Physical activities
            'gym': ['energetic', 'motivated', 'strong', 'determined'],
            'running': ['energetic', 'accomplished', 'determined'],
            'workout': ['energetic', 'motivated', 'strong'],
            'exercise': ['energetic', 'motivated', 'determined'],
            'yoga': ['calm', 'peaceful', 'serene', 'focused'],
            'meditation': ['calm', 'peaceful', 'serene', 'mindful'],
            'swimming': ['refreshed', 'energetic', 'accomplished'],
            'cycling': ['energetic', 'accomplished', 'determined'],
            'hiking': ['energetic', 'accomplished', 'awe'],
            
            # Social activities
            'party': ['happy', 'excited', 'energetic', 'social'],
            'dinner': ['content', 'happy', 'social'],
            'meeting': ['focused', 'productive', 'attentive'],
            'conversation': ['engaged', 'interested', 'connected'],
            'date': ['excited', 'happy', 'romantic'],
            'family': ['happy', 'content', 'connected'],
            'friends': ['happy', 'excited', 'connected'],
            
            # Work and productivity
            'work': ['focused', 'productive', 'determined'],
            'study': ['focused', 'determined', 'attentive'],
            'reading': ['focused', 'calm', 'engaged'],
            'writing': ['focused', 'creative', 'engaged'],
            'coding': ['focused', 'productive', 'determined'],
            'project': ['focused', 'productive', 'determined'],
            'meeting': ['focused', 'attentive', 'productive'],
            
            # Leisure activities
            'movie': ['relaxed', 'entertained', 'engaged'],
            'music': ['relaxed', 'happy', 'moved'],
            'game': ['excited', 'engaged', 'competitive'],
            'shopping': ['excited', 'happy', 'satisfied'],
            'travel': ['excited', 'adventurous', 'awe'],
            'vacation': ['relaxed', 'happy', 'content'],
            'holiday': ['happy', 'excited', 'content'],
            
            # Relaxation and mindfulness
            'relaxation': ['calm', 'peaceful', 'serene'],
            'mindfulness': ['calm', 'focused', 'present'],
            'breathing': ['calm', 'focused', 'present'],
            'spa': ['relaxed', 'calm', 'peaceful'],
            'massage': ['relaxed', 'calm', 'peaceful'],
            
            # Daily activities
            'coffee': ['energetic', 'focused', 'awake'],
            'breakfast': ['energetic', 'fresh', 'optimistic'],
            'lunch': ['satisfied', 'content', 'refreshed'],
            'dinner': ['satisfied', 'content', 'relaxed'],
            'shower': ['refreshed', 'clean', 'energetic'],
            'sleep': ['tired', 'calm', 'peaceful'],
            
            # Temporal context
            'morning': ['energetic', 'fresh', 'optimistic'],
            'afternoon': ['focused', 'productive', 'engaged'],
            'evening': ['tired', 'relaxed', 'content'],
            'night': ['tired', 'calm', 'reflective'],
            'weekend': ['relaxed', 'happy', 'content'],
            'monday': ['determined', 'focused', 'productive'],
            'friday': ['excited', 'happy', 'relaxed']
        }
        
        # Common emotional phrases and metaphorical expressions
        self.emotional_phrases = {
            # Common metaphorical expressions
            'rollercoaster ride': ['excited', 'anxious', 'mixed emotions'],
            'on top of the world': ['happy', 'accomplished', 'proud'],
            'down in the dumps': ['sad', 'depressed', 'disappointed'],
            'cloud nine': ['happy', 'excited', 'elated'],
            'over the moon': ['happy', 'excited', 'elated'],
            'heart of gold': ['kind', 'generous', 'empathy'],
            'heart of stone': ['cold', 'unfeeling', 'indifferent'],
            'walking on air': ['happy', 'excited', 'elated'],
            'feeling blue': ['sad', 'depressed', 'down'],
            'seeing red': ['angry', 'furious', 'enraged'],
            'green with envy': ['jealous', 'envious', 'resentful'],
            'white as a sheet': ['scared', 'frightened', 'terrified'],
            
            # Common emotional phrases
            'feeling pumped': ['energetic', 'excited', 'motivated'],
            'keeping me up': ['anxious', 'worried', 'stressed'],
            'making me feel alive': ['energetic', 'excited', 'happy'],
            'can\'t stop smiling': ['happy', 'joyful', 'content'],
            'on edge': ['anxious', 'nervous', 'stressed'],
            'at peace': ['calm', 'peaceful', 'serene'],
            'in the zone': ['focused', 'productive', 'determined'],
            'in a funk': ['sad', 'depressed', 'down'],
            'in a rut': ['bored', 'stuck', 'frustrated'],
            'on cloud nine': ['happy', 'excited', 'elated'],
            'under the weather': ['tired', 'sick', 'unwell'],
            'feeling blessed': ['grateful', 'thankful', 'happy'],
            'feeling grateful': ['grateful', 'thankful', 'content'],
            'feeling overwhelmed': ['stressed', 'anxious', 'overwhelmed'],
            'feeling accomplished': ['proud', 'accomplished', 'satisfied'],
            'feeling proud': ['proud', 'accomplished', 'satisfied'],
            'feeling confident': ['confident', 'assured', 'strong'],
            'feeling insecure': ['anxious', 'nervous', 'uncertain'],
            'feeling loved': ['loved', 'appreciated', 'happy'],
            'feeling lonely': ['lonely', 'isolated', 'sad'],
            'feeling connected': ['connected', 'engaged', 'happy'],
            'feeling disconnected': ['lonely', 'isolated', 'disconnected'],
            'feeling inspired': ['inspired', 'motivated', 'excited'],
            'feeling motivated': ['motivated', 'determined', 'focused'],
            'feeling unmotivated': ['unmotivated', 'lazy', 'tired'],
            'feeling refreshed': ['refreshed', 'energetic', 'awake'],
            'feeling exhausted': ['tired', 'exhausted', 'drained'],
            'feeling energized': ['energetic', 'awake', 'motivated'],
            'feeling drained': ['tired', 'exhausted', 'drained'],
            'feeling relaxed': ['relaxed', 'calm', 'peaceful'],
            'feeling stressed': ['stressed', 'anxious', 'worried'],
            'feeling calm': ['calm', 'peaceful', 'serene'],
            'feeling anxious': ['anxious', 'nervous', 'worried'],
            'feeling peaceful': ['peaceful', 'calm', 'serene'],
            'feeling worried': ['worried', 'anxious', 'concerned'],
            'feeling serene': ['serene', 'calm', 'peaceful'],
            'feeling tense': ['tense', 'stressed', 'anxious'],
            'feeling mindful': ['mindful', 'present', 'focused'],
            'feeling present': ['present', 'mindful', 'focused'],
            'feeling distracted': ['distracted', 'unfocused', 'scattered'],
            'feeling focused': ['focused', 'attentive', 'determined'],
            'feeling scattered': ['scattered', 'unfocused', 'distracted'],
            'feeling attentive': ['attentive', 'focused', 'engaged'],
            'feeling engaged': ['engaged', 'interested', 'attentive'],
            'feeling disengaged': ['disengaged', 'bored', 'uninterested'],
            'feeling interested': ['interested', 'engaged', 'curious'],
            'feeling uninterested': ['uninterested', 'bored', 'disengaged'],
            'feeling curious': ['curious', 'interested', 'engaged'],
            'feeling bored': ['bored', 'uninterested', 'disengaged']
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
            'wonder': ['wondering', 'amazed', 'astonished', 'awestruck', 'marveling', 'fascinated'],
            
            # Activity-based emotions
            'energetic': ['energetic', 'vibrant', 'lively', 'dynamic', 'powerful', 'strong', 'active'],
            'motivated': ['motivated', 'driven', 'determined', 'focused', 'ambitious', 'goal-oriented'],
            'accomplished': ['accomplished', 'achieved', 'successful', 'triumphant', 'victorious', 'proud'],
            'focused': ['focused', 'concentrated', 'attentive', 'mindful', 'present', 'engaged'],
            'productive': ['productive', 'efficient', 'effective', 'accomplished', 'successful'],
            'relaxed': ['relaxed', 'calm', 'peaceful', 'serene', 'tranquil', 'at ease'],
            'social': ['social', 'connected', 'engaged', 'interactive', 'communicative', 'friendly'],
            'connected': ['connected', 'engaged', 'involved', 'interactive', 'social', 'united'],
            'mindful': ['mindful', 'present', 'aware', 'conscious', 'attentive', 'focused'],
            'present': ['present', 'mindful', 'aware', 'conscious', 'attentive', 'focused']
        }

    def detect_activities(self, text):
        """Detect activities mentioned in the text and map them to emotions."""
        detected_emotions = defaultdict(float)
        
        # Check for activities
        for activity, emotions in self.activity_emotions.items():
            if activity.lower() in text.lower():
                for emotion in emotions:
                    detected_emotions[emotion] += 0.5  # Increased weight for activities
                    
        return detected_emotions
    
    def detect_emotional_phrases(self, text):
        """Detect common emotional phrases in the text."""
        detected_emotions = defaultdict(float)
        
        # Check for emotional phrases
        for phrase, emotions in self.emotional_phrases.items():
            if phrase.lower() in text.lower():
                for emotion in emotions:
                    detected_emotions[emotion] += 0.5  # Increased weight for phrases
                    
        return detected_emotions
    
    def detect_temporal_context(self, text):
        """Detect temporal context in the text."""
        detected_emotions = defaultdict(float)
        
        # Check for temporal context
        for context, emotions in self.activity_emotions.items():
            if context.lower() in text.lower():
                for emotion in emotions:
                    detected_emotions[emotion] += 0.3  # Moderate weight for temporal context
                    
        return detected_emotions

    def analyze_text(self, text):
        """
        Analyze the emotional content of the text using multiple models and approaches.
        """
        # First check for activities and phrases to prioritize them
        activity_emotions = self.detect_activities(text)
        phrase_emotions = self.detect_emotional_phrases(text)
        temporal_emotions = self.detect_temporal_context(text)
        
        # If we have strong activity or phrase matches, prioritize them
        has_strong_context = (len(activity_emotions) > 0 or len(phrase_emotions) > 0)
        
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
        
        # Weight for each model - adjusted to prioritize activities and phrases
        weights = {
            'roberta': 0.2 if has_strong_context else 0.3,  # Reduce weight if we have strong context
            'bert': 0.1 if has_strong_context else 0.2,     # Reduce weight if we have strong context
            'goemotions': 0.1 if has_strong_context else 0.2, # Reduce weight if we have strong context
            'nltk': 0.1,     # Lexicon-based approach
            'activities': 0.3 if has_strong_context else 0.1, # Increase weight if we have strong context
            'phrases': 0.2 if has_strong_context else 0.1    # Increase weight if we have strong context
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
        
        # Add activity-based emotions
        for emotion, score in activity_emotions.items():
            combined_scores[emotion] += score * weights['activities']
        
        # Add emotional phrases
        for emotion, score in phrase_emotions.items():
            combined_scores[emotion] += score * weights['phrases']
            
        # Add temporal context
        for emotion, score in temporal_emotions.items():
            combined_scores[emotion] += score * 0.1  # Lower weight for temporal context
        
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
                'models': {k: v for k, v in results.items()},
                'activities': dict(activity_emotions),
                'phrases': dict(phrase_emotions)
            }
        }
        
        return result 