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

        # Activity detection keywords
        self.activity_keywords = {
            'gym': ['gym', 'workout', 'exercise', 'training', 'fitness', 'lifting', 'weights'],
            'coffee': ['coffee', 'caffeine', 'espresso', 'latte', 'cappuccino', 'brew'],
            'running': ['running', 'jogging', 'sprint', 'marathon', 'track'],
            'yoga': ['yoga', 'meditation', 'stretching', 'mindfulness'],
            'swimming': ['swimming', 'pool', 'dive', 'swim'],
            'cycling': ['cycling', 'bike', 'bicycle', 'ride'],
            'hiking': ['hiking', 'trail', 'mountain', 'walk'],
            'party': ['party', 'celebration', 'festival', 'gathering'],
            'dinner': ['dinner', 'meal', 'restaurant', 'food'],
            'meeting': ['meeting', 'conference', 'presentation', 'discussion'],
            'conversation': ['conversation', 'talk', 'chat', 'discussion'],
            'date': ['date', 'romantic', 'dinner', 'movie'],
            'family': ['family', 'parents', 'siblings', 'relatives'],
            'friends': ['friends', 'buddies', 'pals', 'mates'],
            'work': ['work', 'office', 'job', 'career'],
            'study': ['study', 'learning', 'education', 'school'],
            'reading': ['reading', 'book', 'novel', 'article'],
            'writing': ['writing', 'author', 'compose', 'draft'],
            'coding': ['coding', 'programming', 'develop', 'code'],
            'project': ['project', 'assignment', 'task', 'work'],
            'movie': ['movie', 'film', 'cinema', 'theater'],
            'music': ['music', 'song', 'tune', 'melody'],
            'game': ['game', 'play', 'gaming', 'sport'],
            'shopping': ['shopping', 'buy', 'purchase', 'mall'],
            'travel': ['travel', 'trip', 'journey', 'vacation'],
            'vacation': ['vacation', 'holiday', 'break', 'leave'],
            'holiday': ['holiday', 'festival', 'celebration', 'break'],
            'relaxation': ['relaxation', 'rest', 'chill', 'unwind'],
            'mindfulness': ['mindfulness', 'meditation', 'awareness', 'presence'],
            'breathing': ['breathing', 'breath', 'inhale', 'exhale'],
            'spa': ['spa', 'massage', 'treatment', 'therapy'],
            'massage': ['massage', 'therapy', 'relax', 'treatment'],
            'breakfast': ['breakfast', 'morning', 'meal', 'eat'],
            'lunch': ['lunch', 'noon', 'meal', 'eat'],
            'dinner': ['dinner', 'evening', 'meal', 'eat'],
            'shower': ['shower', 'bath', 'clean', 'wash'],
            'sleep': ['sleep', 'bed', 'rest', 'nap']
        }

        # Activity-based emotion mappings
        self.activity_emotions = {
            'gym': ['motivated', 'energetic', 'excited', 'determined', 'strong', 'focused'],
            'coffee': ['energetic', 'focused', 'awake', 'alert', 'productive'],
            'running': ['energetic', 'accomplished', 'determined', 'motivated', 'strong'],
            'yoga': ['calm', 'peaceful', 'serene', 'focused', 'mindful'],
            'swimming': ['refreshed', 'energetic', 'accomplished', 'relaxed'],
            'cycling': ['energetic', 'accomplished', 'determined', 'free'],
            'hiking': ['energetic', 'accomplished', 'awe', 'connected'],
            'party': ['happy', 'excited', 'energetic', 'social', 'joyful'],
            'dinner': ['content', 'happy', 'social', 'satisfied'],
            'meeting': ['focused', 'productive', 'attentive', 'engaged'],
            'conversation': ['engaged', 'interested', 'connected', 'social'],
            'date': ['excited', 'happy', 'romantic', 'nervous'],
            'family': ['happy', 'content', 'connected', 'loved'],
            'friends': ['happy', 'excited', 'connected', 'social'],
            'work': ['focused', 'productive', 'determined', 'motivated'],
            'study': ['focused', 'determined', 'attentive', 'motivated'],
            'reading': ['focused', 'calm', 'engaged', 'interested'],
            'writing': ['focused', 'creative', 'engaged', 'productive'],
            'coding': ['focused', 'productive', 'determined', 'creative'],
            'project': ['focused', 'productive', 'determined', 'motivated'],
            'movie': ['relaxed', 'entertained', 'engaged', 'interested'],
            'music': ['relaxed', 'happy', 'moved', 'emotional'],
            'game': ['excited', 'engaged', 'competitive', 'focused'],
            'shopping': ['excited', 'happy', 'satisfied', 'energetic'],
            'travel': ['excited', 'adventurous', 'awe', 'curious'],
            'vacation': ['relaxed', 'happy', 'content', 'excited'],
            'holiday': ['happy', 'excited', 'content', 'joyful'],
            'relaxation': ['calm', 'peaceful', 'serene', 'relaxed'],
            'mindfulness': ['calm', 'focused', 'present', 'aware'],
            'breathing': ['calm', 'focused', 'present', 'relaxed'],
            'spa': ['relaxed', 'calm', 'peaceful', 'serene'],
            'massage': ['relaxed', 'calm', 'peaceful', 'serene'],
            'breakfast': ['energetic', 'fresh', 'optimistic', 'ready'],
            'lunch': ['satisfied', 'content', 'refreshed', 'energetic'],
            'dinner': ['satisfied', 'content', 'relaxed', 'social'],
            'shower': ['refreshed', 'clean', 'energetic', 'awake'],
            'sleep': ['tired', 'calm', 'peaceful', 'relaxed']
        }
        
        # Emotion categories with expanded definitions
        self.emotion_categories = {
    'positive': [
        # Original emotions
        'joy', 'love', 'hope', 'gratitude', 'pride', 'contentment', 'excitement',
                        'relief', 'curiosity', 'inspiration', 'optimism', 'trust', 'wonder',
                        'amusement', 'awe', 'desire', 'empathy', 'enthusiasm', 'gratitude',
                        'happiness', 'interest', 'joy', 'love', 'optimism', 'pride', 'relief',
        'satisfaction', 'serenity', 'surprise', 'trust',

        # Activities
        'gym', 'running', 'workout', 'exercise', 'swimming', 'cycling', 'hiking',
        'party', 'dinner', 'meeting', 'conversation', 'date', 'family', 'friends',
        'work', 'study', 'reading', 'writing', 'coding', 'project',
        'movie', 'music', 'game', 'shopping', 'travel', 'vacation', 'holiday',
        'coffee', 'breakfast', 'lunch', 'shower',

        # Temporal contexts
        'morning', 'afternoon', 'weekend', 'monday', 'friday',

        # Metaphorical expressions
        'on top of the world', 'cloud nine', 'over the moon', 'heart of gold',
        'walking on air',

        # Phrases
        "can't stop smiling", 'feeling pumped', 'making me feel alive', 'in the zone',
        'on cloud nine', 'feeling blessed', 'feeling grateful', 'feeling accomplished',
        'feeling proud', 'feeling confident', 'feeling loved', 'feeling connected',
        'feeling inspired', 'feeling motivated', 'feeling refreshed', 'feeling energized',
        'rollercoaster ride'
    ],

    'negative': [
        # Original emotions
        'sadness', 'anger', 'fear', 'disappointment', 'confusion', 'anxiety', 'boredom',
        'embarrassment', 'envy', 'guilt', 'loneliness', 'pessimism', 'regret',
        'shame', 'annoyance', 'disgust', 'frustration', 'hate', 'irritation',
        'jealousy', 'nervousness', 'stress', 'tension', 'worry',

        # Metaphorical expressions
        'down in the dumps', 'heart of stone', 'feeling blue', 'seeing red',
        'green with envy', 'white as a sheet',

        # Phrases
        'keeping me up', 'on edge', 'in a funk', 'in a rut', 'under the weather',
        'feeling overwhelmed', 'feeling insecure', 'feeling lonely',
        'feeling disconnected', 'feeling unmotivated', 'feeling exhausted',
        'feeling drained', 'feeling stressed', 'feeling anxious', 'feeling worried',
        'feeling tense', 'feeling distracted', 'feeling scattered',
        'feeling disengaged', 'feeling uninterested', 'feeling bored', 'rollercoaster ride'
    ],

    'neutral': [
        # Original emotions
        'neutral', 'contemplation', 'nostalgia', 'determination', 'anticipation',
        'attention', 'calmness', 'concentration', 'meditation', 'observation',
        'patience', 'reflection', 'thoughtfulness',

        # Activities & mindfulness
        'yoga', 'meditation', 'relaxation', 'mindfulness', 'breathing', 'spa',
        'massage', 'sleep',

        # Temporal contexts
        'evening', 'night',

        # Metaphorical expressions
        'rollercoaster ride', 'at peace',

        # Phrases
        'feeling relaxed', 'feeling calm', 'feeling peaceful', 'feeling serene',
        'feeling present'
    ]
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
            'present': ['present', 'mindful', 'aware', 'conscious', 'attentive', 'focused'],

            #Activities
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
            'friday': ['excited', 'happy', 'relaxed'],
        
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
        
    

    def detect_activity(self, text):
        """
        Detect if the text contains any activity keywords and return the associated emotions.
        """
        text_lower = text.lower()
        detected_activities = []
        
        for activity, keywords in self.activity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_activities.append(activity)
        
        return detected_activities

    def analyze_text(self, text):
        """
        Analyze the emotional content of the text using multiple models and approaches.
        """
        # First check for activities
        detected_activities = self.detect_activity(text)
        
        # If activities are detected, prioritize their associated emotions
        if detected_activities:
            activity_emotions = []
            for activity in detected_activities:
                if activity in self.activity_emotions:
                    activity_emotions.extend(self.activity_emotions[activity])
            
            # Remove duplicates and create emotion objects
            unique_emotions = list(set(activity_emotions))
            detected_emotions = []
            
            # Assign high confidence to activity-based emotions
            for emotion in unique_emotions:
                detected_emotions.append({
                    'emotion': emotion,
                    'confidence': 0.9  # High confidence for activity-based emotions
                })
            
            # Get keywords for all detected emotions
            all_keywords = []
            for emotion_data in detected_emotions:
                emotion = emotion_data['emotion']
                if emotion in self.emotion_keywords:
                    all_keywords.extend(self.emotion_keywords[emotion])
            
            # Remove duplicates
            all_keywords = list(set(all_keywords))
            
            return {
                'primary_emotion': detected_emotions[0]['emotion'] if detected_emotions else 'neutral',
                'emotions': detected_emotions,
                'keywords': all_keywords,
                'activities': detected_activities,
                'raw_scores': {
                    'activity_based': True
                }
            }
        
        # If no activities detected, proceed with normal emotion analysis
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
            'activities': [],
            'raw_scores': {
                'nltk': nltk_scores,
                'models': {k: v for k, v in results.items()}
            }
        }
        
        return result 