
# AI Powered Emotion Detector

![Project Banner](https://raw.githubusercontent.com/Param1304/AI_Powered_Emotion_Detector/main/Second_Project_Zidio/banner.png)

AI Powered Emotion Detector is an end-to-end, Django-based web application designed to assess human emotions through multiple modalities – **text**, **facial expressions**, and **voice**. Developed as part of my Data Science Internship at Zidio Development, this project leverages state-of-the-art machine learning and computer vision techniques to provide real-time mood detection, analysis, and task optimization.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Demo & Screenshots](#demo--screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

In today's fast-paced work environments, understanding employee sentiment can be key to enhancing productivity and well-being. This project provides a holistic solution by:

- Analyzing **text inputs** with a fine-tuned BERT model for mood classification.
- Detecting **facial expressions** using OpenCV’s Haar cascades (face, smile, and eye detection) to infer emotions in real time.
- Implementing **voice emotion detection** using Librosa to extract audio features from 5-second recordings and classify them into emotion categories.
- Presenting the aggregated data through an interactive **analysis dashboard** built with Plotly.js for visual insights into mood trends.

---

## Features

- **Multi-modal Emotion Detection**
  - **Text-based Analysis**: Leverages a pretrained BERT model with cosine similarity to classify user input into categories such as Normal, Anxiety, Depression, or Suicidal.
  - **Facial Emotion Recognition**: Utilizes OpenCV for real-time detection of facial expressions, employing cascades for face, smile, and eye detection.
  - **Voice Emotion Detection**: Records 5 seconds of audio via the browser, extracts features (MFCC, chroma, spectral contrast) with Librosa, and classifies emotion.
- **Interactive Dashboard**
  - Visualize mood trends over time using a **Line Chart**.
  - View mood distribution across different categories using a **Bar Chart**.
- **Task Optimization Engine**
  - Provides personalized task recommendations based on the detected emotional state.
- **End-to-End Web Application**
  - Full-stack solution built with Django, integrating AI models, computer vision, and data analytics for a seamless user experience.

---

## Project Structure

```
AI_Powered_Emotion_Detector/
└── Second_Project_Zidio/
    ├── myapp/
    │   ├── migrations/
    │   ├── static/
    │   │   ├── analysis.css         # Styles for the dashboard
    │   │   ├── detect_face.css      # Styles for facial emotion UI
    │   │   ├── record_audio.js      # JavaScript for recording audio
    │   │   └── styles.css           # General styling
    │   ├── templates/
    │   │   ├── analyse_data.html    # Dashboard for mood analysis
    │   │   ├── detect_face.html     # Facial emotion detection UI
    │   │   ├── detect_mood.html     # Text-based mood detection UI
    │   │   ├── detect_voice.html    # Voice emotion detection UI
    │   │   ├── home.html            # Homepage
    │   │   └── suggest_task.html    # Task suggestion page
    │   ├── __init__.py
    │   ├── admin.py                 # Django admin config
    │   ├── apps.py                  # App configuration
    │   ├── models.py                # Database models (e.g., MoodEntry)
    │   ├── tests.py
    │   ├── urls.py                  # App-level URL routing
    │   └── views.py                 # View functions for all features
    ├── task_optimizer/
    │   ├── __init__.py
    │   ├── asgi.py
    │   ├── settings.py              # Django project settings
    │   ├── urls.py                  # Project-level URL routing
    │   └── wsgi.py
    ├── db.sqlite3                   # SQLite database
    ├── manage.py                    # Django management script
    └── requirements.txt             # Project dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- Virtualenv (recommended)
- Git

### Setup Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Param1304/AI_Powered_Emotion_Detector.git
   cd AI_Powered_Emotion_Detector/Second_Project_Zidio
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   # Activate on Windows
   venv\Scripts\activate
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Database Migrations**

   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start the Django Server**

   ```bash
   python manage.py runserver
   ```

6. **Access the Application**

   Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Usage

- **Home Page**: Provides navigation to various features.
- **Detect Mood**: Enter text responses to analyze your mood.
- **Detect Face**: Use your webcam to get real-time facial emotion detection.
- **Detect Voice**: Record a 5-second audio clip to analyze your voice emotion.
- **Analyse Data**: Explore the mood trends and distributions on the interactive dashboard.
- **Suggest Task**: Receive personalized task recommendations based on detected emotions.

---

## Technologies Used

- **Backend**: Django, SQLite
- **Frontend**: HTML, CSS, JavaScript
- **Computer Vision**: OpenCV (Face, Smile, Eye detection)
- **Natural Language Processing**: BERT (via Hugging Face Transformers)
- **Audio Processing**: Librosa, SoundFile, Pydub (if needed)
- **Data Visualization**: Plotly.js
- **Machine Learning**: Scikit-Learn, TensorFlow (if using neural networks)

---

## Demo & Screenshots

### Home Page
![Home Page](https://raw.githubusercontent.com/Param1304/AI_Powered_Emotion_Detector/main/Second_Project_Zidio/screenshots/home.png)

### Facial Emotion Detection
![Facial Detection](https://raw.githubusercontent.com/Param1304/AI_Powered_Emotion_Detector/main/Second_Project_Zidio/screenshots/detect_face.png)

### Voice Emotion Detection
![Voice Detection](https://raw.githubusercontent.com/Param1304/AI_Powered_Emotion_Detector/main/Second_Project_Zidio/screenshots/detect_voice.png)

### Mood Analysis Dashboard
![Dashboard](https://raw.githubusercontent.com/Param1304/AI_Powered_Emotion_Detector/main/Second_Project_Zidio/screenshots/analyse_data.png)

---

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](https://github.com/Param1304/AI_Powered_Emotion_Detector/issues) if you want to contribute.

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Param Singh**  
Data Science Intern at Zidio Development  
[Email: your-email@example.com](mailto:your-email@example.com)  
[LinkedIn](https://www.linkedin.com/in/your-linkedin-profile)

---

Feel free to reach out if you have any questions or feedback!

---

This README provides a detailed and attractive overview of your project, outlining its technical scope, usage, and contribution guidelines in a professional manner. Enjoy showcasing your work!
