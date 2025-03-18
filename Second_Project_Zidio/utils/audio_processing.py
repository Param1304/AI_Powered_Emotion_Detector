import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from django.conf import settings
MODEL_PATH = r"C:\Users\Param\OneDrive\文档\Zidio_Project\Second_Project_Zidio\models\emotion_model.keras"
LABEL_ENCODER_PATH = r"C:\Users\Param\OneDrive\文档\Zidio_Project\Second_Project_Zidio\models\label_encoder.pkl"
class AudioEmotionDetector:
    def __init__(self):
        self.model = load_model(settings.MODEL_PATH)
        self.label_encoder = joblib.load(settings.LABEL_ENCODER_PATH)
        
    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        stft = np.abs(librosa.stft(y))
        features = np.concatenate((
            np.mean(mfccs, axis=1),
            np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1),
            np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1),
            np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)
        ))
        return features

    def predict_emotion(self, audio_path):
        features = self.extract_features(audio_path)
        prediction = self.model.predict(np.expand_dims(features, axis=0))
        return self.label_encoder.inverse_transform([np.argmax(prediction)])[0]