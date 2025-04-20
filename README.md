# 🎵 Soul Sound - Emotion-Based Music Recommender

<div align="center">

![Soul Sound Logo](static/images/SoulSound.png)

*Where emotions meet melodies*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.0+-yellow.svg)](https://www.mongodb.com/)
[![Spotify API](https://img.shields.io/badge/Spotify%20API-Enabled-brightgreen.svg)](https://developer.spotify.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg)](https://huggingface.co/)

</div>

## 🌟 Overview

Soul Sound is an innovative web application that bridges the gap between emotional expression and musical discovery. By analyzing the emotional content of journal entries, it provides personalized song recommendations that resonate with your current mood, creating a unique therapeutic experience through music.

## 🎯 Problem Statement

To create an AI/ML project that can or may help in real life situations

## 🌱 Evolution of the Idea

Initially, we thought of making an AI application called "RizzOrRoast" wherein the user rizzes or roasts an AI competitor and the AI does the same with varying intensity and they're scored based on their attempts and lose or gain points accordingly. We then thought what if even a censored bot does a lot of emotional damage that the person feels suicidal and we're held responsible. HENCE, we took an absolute U-turn and created Soul Sound, an emotion based music recommender, which analyzes the emotions of a person from their journal entry and suggests songs based on how they felt on that day. It stores their journal entry and playlists and calms them with music and visually appealing backgrounds.

## 👥 Track and Contributors

### Track
- Generative AI and machine learning

### Contributors
- [Navya Balaji](https://github.com/navyabalaji) - BT2024262
- [Thapan Komaravelly](https://github.com/SapphireGaze21) - BT2024076
- [Ruthvik CSS](https://github.com/Ruthvik-CSS) - BT2024051

## ✨ Features

### 🎨 User Interface
- Modern, responsive design with Tailwind CSS
- Serene video backgrounds that rotate automatically
- Calming background music, which can be muted
- Intuitive navigation with smooth transitions
- Emotion-based color schemes and visual feedback

### 📝 Journal Management
- Rich text journal entries with emotion analysis
- Timestamp tracking with proper timezone handling
- Journal organization with search functionality
- Secure storage with MongoDB integration

### 🎵 Music Integration
- Spotify song recommendations based on detected emotions
- Emotion-based playlist generation
- One-click song saving to personal playlists
- Comprehensive playlist management system

### 🔒 Security
- Secure user authentication with Flask-Login
- Data encryption for sensitive information
- Privacy-focused design with user data protection

## 🛠️ Technology Stack

### Frontend
- **HTML5/CSS3**: Modern web standards
- **Tailwind CSS**: Utility-first CSS framework
- **JavaScript**: Client-side interactivity
- **Video.js**: Video background handling

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework
- **MongoDB**: NoSQL database
- **PyMongo**: MongoDB driver for Python

### APIs & Services
- **Spotify Web API**: Music recommendations and playback
- **Hugging Face Transformers**: Emotion analysis model
- **Flask-Login**: User authentication

### Development Tools
- **IDE**: Visual Studio Code
- **Version Control**: Git
- **Package Manager**: pip
- **Environment Management**: venv

### Libraries
- **transformers**: Hugging Face's emotion analysis
- **torch**: PyTorch for model inference
- **flask**: Web application framework
- **pymongo**: MongoDB integration
- **python-dotenv**: Environment variable management
- **requests**: HTTP requests handling

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- MongoDB (local or Atlas)
- Spotify Developer Account
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soulsound.git
cd soulsound
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Spotify API credentials:
     ```
     SPOTIFY_CLIENT_ID=your_client_id
     SPOTIFY_CLIENT_SECRET=your_client_secret
     ```

5. Start MongoDB:
   - If using local MongoDB, ensure the service is running
   - If using MongoDB Atlas, ensure your connection string is properly configured

## 🏃‍♂️ How to Run

1. Start the application:
```bash
python app.py
```

2. Access the application at `http://localhost:5000`

3. Register a new account or log in with existing credentials

4. Start journaling and receiving personalized music recommendations!

## 🌐 Applications

Soul Sound has various applications across different domains:

### 🧠 Mental Health
- Emotional self-awareness and regulation
- Therapeutic journaling with music support
- Mood tracking over time

### 🎓 Education
- Emotional intelligence development
- Language learning through emotional expression
- Creative writing with emotional context

### 💼 Professional Development
- Stress management and work-life balance
- Team building through shared emotional experiences
- Productivity enhancement through mood-based music

### 🏥 Healthcare
- Complementary therapy for mental health patients
- Emotional support for patients and caregivers
- Rehabilitation and recovery support

## 🔮 Further Improvements

- **Advanced Emotion Detection**: Implement more sophisticated emotion analysis models
- **User Feedback based suggestion**: Allow user to give feedback on suggested music and improve the recommendations
- **Mobile Application**: Develop native mobile apps for iOS and Android
- **Community Features**: Allow users to share playlists and journal insights
- **AI Music Composition**: Generate original music based on detected emotions
- **Analytics Dashboard**: Provide insights into emotional patterns over time
- **Offline Mode**: Enable journaling and playlist access without internet connection

## 🙏 Acknowledgments

- [Spotify](https://www.spotify.com/) for their amazing API
- [Hugging Face](https://huggingface.co/) for emotion analysis models
- [Tailwind CSS](https://tailwindcss.com/) for the beautiful UI framework
- [MongoDB](https://www.mongodb.com/) for the database solution
- [Flask](https://flask.palletsprojects.com/) for the web framework

## Demo Video

Link to demo video here
(https://drive.google.com/drive/folders/1Fu1y8Pv1vZtxqpQqhgV3qO9XZ_NzhO3O)

---

<div align="center">
Made with ❤️ by the Soul Sound Team
</div> 
