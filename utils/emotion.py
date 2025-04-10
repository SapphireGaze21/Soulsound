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
            # Primary emotions from the model
            'joy': ['happy', 'upbeat', 'cheerful', 'joyful', 'excited'],
            'sadness': ['sad', 'melancholic', 'emotional', 'depressing', 'heartbroken', 'upset'],
            'anger': ['angry', 'intense', 'powerful', 'furious', 'aggressive'],
            'fear': ['calm', 'peaceful', 'relaxing', 'anxious', 'nervous'],
            'love': ['romantic', 'love', 'passionate', 'affectionate', 'tender', 'care'],
            'surprise': ['energetic', 'exciting', 'dynamic', 'amazing', 'wonderful', 'wow'],
            'neutral': ['balanced', 'moderate', 'steady', 'neutral', 'calm', 'peaceful', 'relaxing', 'serene', 'tranquil', 'composed', 'fine'],
            
            # Additional emotions for broader spectrum
            'hope': ['hopeful', 'inspiring', 'uplifting', 'motivational', 'encouraging'],
            'gratitude': ['grateful', 'thankful', 'appreciative', 'blessed', 'fortunate'],
            'pride': ['proud', 'accomplished', 'successful', 'triumphant', 'victorious'],
            'nostalgia': ['nostalgic', 'memories', 'retro', 'vintage', 'classic'],
            'contemplation': ['thoughtful', 'reflective', 'meditative', 'philosophical', 'deep'],
            'determination': ['determined', 'focused', 'driven', 'ambitious', 'resolute'],
            'contentment': ['content', 'satisfied', 'fulfilled', 'at peace', 'serene'],
            'confusion': ['confused', 'uncertain', 'doubtful', 'questioning', 'searching'],
            'disappointment': ['disappointed', 'let down', 'frustrated', 'disheartened', 'discouraged'],
            'excitement': ['excited', 'thrilled', 'enthusiastic', 'eager', 'energetic'],
            
            # Even more emotions for comprehensive analysis
            'anxiety': ['anxious', 'worried', 'nervous', 'uneasy', 'tense'],
            'relief': ['relieved', 'calm', 'peaceful', 'tranquil', 'serene'],
            'boredom': ['bored', 'uninterested', 'monotonous', 'tedious', 'dull'],
            'curiosity': ['curious', 'interested', 'inquisitive', 'exploratory', 'fascinated'],
            'embarrassment': ['embarrassed', 'ashamed', 'self-conscious', 'awkward', 'uncomfortable'],
            'envy': ['envious', 'jealous', 'covetous', 'resentful', 'begrudging'],
            'guilt': ['guilty', 'remorseful', 'regretful', 'apologetic', 'contrite'],
            'inspiration': ['inspired', 'creative', 'imaginative', 'innovative', 'visionary'],
            'loneliness': ['lonely', 'isolated', 'alone', 'solitary', 'forsaken'],
            'optimism': ['optimistic', 'positive', 'hopeful', 'confident', 'upbeat'],
            'pessimism': ['pessimistic', 'negative', 'cynical', 'doubtful', 'skeptical'],
            'regret': ['regretful', 'remorseful', 'sorry', 'apologetic', 'contrite'],
            'shame': ['ashamed', 'embarrassed', 'humiliated', 'disgraced', 'mortified'],
            'trust': ['trusting', 'faithful', 'loyal', 'reliable', 'dependable'],
            'wonder': ['wondering', 'amazed', 'astonished', 'awestruck', 'marveling']
        }
        
        # Emotion categories for grouping similar emotions
        self.emotion_categories = {
            'positive': ['joy', 'love', 'hope', 'gratitude', 'pride', 'contentment', 'excitement', 
                        'relief', 'curiosity', 'inspiration', 'optimism', 'trust', 'wonder'],
            'negative': ['sadness', 'anger', 'fear', 'disappointment', 'confusion', 
                        'anxiety', 'boredom', 'embarrassment', 'envy', 'guilt', 'loneliness', 
                        'pessimism', 'regret', 'shame'],
            'neutral': ['neutral', 'contemplation', 'nostalgia', 'determination', 'surprise']
        }
        
        # Emotion similarity groups for detecting repeated emotions
        self.emotion_similarity = {
            'joy': ['excitement', 'optimism', 'hope', 'contentment'],
            'sadness': ['disappointment', 'loneliness', 'regret', 'pessimism'],
            'anger': ['frustration', 'irritation', 'rage', 'resentment'],
            'fear': ['anxiety', 'worry', 'nervousness', 'unease'],
            'love': ['affection', 'tenderness', 'fondness', 'attachment'],
            'surprise': ['astonishment', 'amazement', 'wonder', 'awe'],
            'neutral': ['calm', 'balanced', 'steady', 'composed']
        }

    def analyze_text(self, text):
        """
        Analyze the emotional content of the text and return all emotions
        with their confidence scores and associated keywords for Spotify search.
        """
        # Debug: Print input text
        print(f"Analyzing text: {text[:100]}...")
        
        results = self.classifier(text)[0]
        
        # Debug: Print raw results
        print(f"Raw emotion results: {results}")
        
        # Sort emotions by confidence score
        sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Get all emotions with confidence above 0.1
        detected_emotions = []
        for emotion in sorted_emotions:
            if emotion['score'] > 0.1:  # Only include emotions with significant confidence
                emotion_label = emotion['label'].lower()
                detected_emotions.append({
                    'emotion': emotion_label,
                    'confidence': emotion['score']
                })
        
        # If no emotions detected with confidence > 0.1, use the top emotion
        if not detected_emotions and sorted_emotions:
            top_emotion = sorted_emotions[0]
            detected_emotions.append({
                'emotion': top_emotion['label'].lower(),
                'confidence': top_emotion['score']
            })
        
        # Check for repeated or similar emotions
        if len(detected_emotions) > 1:
            # Group similar emotions
            grouped_emotions = []
            processed_emotions = set()
            
            for emotion_data in detected_emotions:
                emotion = emotion_data['emotion']
                
                # Skip if already processed
                if emotion in processed_emotions:
                    continue
                
                # Find similar emotions
                similar_emotions = []
                for primary, similar_list in self.emotion_similarity.items():
                    if emotion == primary or emotion in similar_list:
                        similar_emotions = [primary] + similar_list
                        break
                
                # Group similar emotions
                group = []
                for sim_emotion in similar_emotions:
                    for emotion_data2 in detected_emotions:
                        if emotion_data2['emotion'] == sim_emotion and sim_emotion not in processed_emotions:
                            group.append(emotion_data2)
                            processed_emotions.add(sim_emotion)
                
                if group:
                    # Calculate average confidence for the group
                    total_confidence = sum(e['confidence'] for e in group)
                    avg_confidence = total_confidence / len(group)
                    
                    # Use the emotion with highest confidence as the group representative
                    group.sort(key=lambda x: x['confidence'], reverse=True)
                    representative = group[0]
                    
                    # Add the grouped emotion with the average confidence
                    grouped_emotions.append({
                        'emotion': representative['emotion'],
                        'confidence': avg_confidence,
                        'similar_emotions': [e['emotion'] for e in group[1:]]
                    })
            
            # Add any remaining emotions that weren't grouped
            for emotion_data in detected_emotions:
                if emotion_data['emotion'] not in processed_emotions:
                    grouped_emotions.append(emotion_data)
            
            # Replace detected_emotions with grouped_emotions
            detected_emotions = grouped_emotions
        
        # Debug: Print detected emotions
        print(f"Detected emotions: {detected_emotions}")
        
        # Get keywords for all detected emotions
        all_keywords = []
        for emotion_data in detected_emotions:
            emotion = emotion_data['emotion']
            if emotion in self.emotion_keywords:
                all_keywords.extend(self.emotion_keywords[emotion])
        
        # Remove duplicates
        all_keywords = list(set(all_keywords))
        
        # Determine the primary emotion (highest confidence)
        primary_emotion = detected_emotions[0]['emotion'] if detected_emotions else 'neutral'
        
        result = {
            'primary_emotion': primary_emotion,
            'emotions': detected_emotions,
            'keywords': all_keywords
        }
        
        # Debug: Print final result
        print(f"Final emotion result: {result}")
        
        return result 