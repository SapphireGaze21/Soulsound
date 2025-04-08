from transformers import pipeline

class EmotionAnalyzer:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Map emotions to Spotify search keywords
        self.emotion_keywords = {
            'joy': ['happy', 'upbeat', 'cheerful'],
            'sadness': ['sad', 'melancholic', 'emotional'],
            'anger': ['angry', 'intense', 'powerful'],
            'fear': ['calm', 'peaceful', 'relaxing'],
            'love': ['romantic', 'love', 'passionate'],
            'surprise': ['energetic', 'exciting', 'dynamic'],
            'neutral': ['balanced', 'moderate', 'steady']
        }

    def analyze_text(self, text):
        """
        Analyze the emotional content of the text and return the primary emotion
        and associated keywords for Spotify search.
        """
        results = self.classifier(text)[0]
        
        # Get the emotion with the highest score
        primary_emotion = max(results, key=lambda x: x['score'])
        emotion_label = primary_emotion['label'].lower()
        
        # Get keywords for the detected emotion
        keywords = self.emotion_keywords.get(emotion_label, ['neutral'])
        
        return {
            'emotion': emotion_label,
            'confidence': primary_emotion['score'],
            'keywords': keywords
        } 