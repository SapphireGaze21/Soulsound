# Emotion-Based Music Recommender

A web application that analyzes the emotional content of journal entries and recommends songs based on the detected mood using Spotify.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Spotify API credentials:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new application
   - Copy the Client ID and Client Secret
   - Create a `.env` file in the project root with:
     ```
     SPOTIFY_CLIENT_ID=your_client_id
     SPOTIFY_CLIENT_SECRET=your_client_secret
     ```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Features

- Journal entry input with emotion analysis
- Emotion detection using Hugging Face's pre-trained model
- Spotify song recommendations based on detected emotions
- Clickable Spotify links for recommended songs

## Project Structure

```
.
├── app.py              # Main application file
├── static/            # Static files (CSS, JS)
├── templates/         # HTML templates
├── utils/            # Utility functions
│   ├── emotion.py    # Emotion analysis functions
│   └── spotify.py    # Spotify API integration
└── requirements.txt   # Project dependencies
``` 