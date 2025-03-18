from django.shortcuts import render
from django.http import JsonResponse
import soundfile as sf
import librosa
import os
import joblib
# Create your views here.
def home(request):
    return render(request, 'home.html', {'mood':'Normal'})

import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from .models import MoodEntry
MODEL_PATH = r"C:\Users\Param\OneDrive\文档\Zidio_Project\saved_mental_status_bert"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()
reference_sentences = {
    "Normal": "I feel okay today, nothing much to complain about.",
    "Normal":"I am motivated to work. No issues",
    "Normal":"No I am fine. I am okay",
    "Normal":"I am well to do",
    "Depression": "I feel like I am at the end, nothing I do is ever right.",
    "Depression": "I hardly find anything enjoyable or pleasurable",
    "Depression":"Everything is so bad",
    "Depression": "I am feeling Isolated and left over",
    "Suicidal": "I have given up on life. I wish everything would just end.",
    "Suicidal":"I have nothing to look onto. Everything is finished",
    "Suicidal":"I quit. I cannot continue anymore",
    "Anxiety": "I am really worried, I can't seem to relax.",
    "Anxiety":"I am sad and anxious. I am frustrated",
    "Anxiety":"Many time I canno control my thoughts. I do overthinking.",
    "Anxiety":"I cannot take decisions. I am feeling weak",
}

reference_inputs = tokenizer(list(reference_sentences.values()), return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    reference_outputs = model(**reference_inputs)
reference_embeddings = reference_outputs.last_hidden_state[:, 0, :].numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_mood(request):
    if request.method == "POST":
        responses = [request.POST[f"q{i}"] for i in range(1, 6)]

        # Process responses with BERT
        inputs = tokenizer(responses, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Compare responses with reference embeddings
        predicted_labels = []
        for emb in sentence_embeddings:
            similarities = {label: cosine_similarity(emb, ref_emb) for label, ref_emb in zip(reference_sentences.keys(), reference_embeddings)}
            predicted_label = max(similarities, key=similarities.get)
            predicted_labels.append(predicted_label)

        # Determine the most common mood
        final_mood = max(set(predicted_labels), key=predicted_labels.count)
        MoodEntry.objects.create(text=" | ".join(responses), mood=final_mood)
        return render(request, "detect_mood.html", {"mood": final_mood})
    return render(request, "detect_mood.html")

def suggest_task(request,mood):
    tasks=[]
    if mood=="Normal":
        tasks=[
            "Start working on your most important project of the day.",
            "Take a short break, and then plan your day ahead.",
            "Meet with a colleague to discuss collaborative work."
        ]
    elif mood == "Depression":
        tasks = [
            "Take a walk outside to clear your mind.",
            "Try writing down your thoughts in a journal.",
            "Start with a small, achievable task like making a to-do list."
        ]
    elif mood == "Suicidal":
        tasks = [
            "Reach out to a mental health professional.",
            "Contact a friend or family member for support.",
            "Take a break and try to rest. Avoid overloading yourself."
        ]
    elif mood == "Anxiety":
        tasks = [
            "Practice deep breathing exercises for a few minutes.",
            "Organize your workspace to reduce stress.",
            "Take a break and listen to calming music."
        ]
    return render(request, "suggest_task.html", {"tasks": tasks, "mood": mood})

from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import numpy as np

# Load OpenCV models for face, eyes, and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


recent_emotions = []

# Emotion classification based on face brightness, smile, and eye detection
def classify_emotion(gray_face, face_region_color):
    global recent_emotions
    mean_intensity = np.mean(gray_face)
    smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
    # Decision rules 
    if len(smiles) > 0:
        emotion = "Happy (Normal)"
    elif len(eyes) == 0: 
        emotion = "Depression"
    elif mean_intensity > 100:
        emotion = "Normal"
    elif 130 < mean_intensity <= 180:
        emotion = "Anxiety"
    elif 80 < mean_intensity <= 130:
        emotion = "Depression"
    else:
        emotion = "Suicidal"

    recent_emotions.append(emotion)
    if len(recent_emotions) > 5:
        recent_emotions.pop(0)

    return max(set(recent_emotions), key=recent_emotions.count)  # Most frequent recent emotion

def generate_frames():
    cap = cv2.VideoCapture(0)  
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region_gray = gray[y:y + h, x:x + w]
            face_region_color = frame[y:y + h, x:x + w]
            emotion = classify_emotion(face_region_gray, face_region_color)

            # Draw bounding box for face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Eye detection within the face region
            eyes = eye_cascade.detectMultiScale(face_region_gray, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_region_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

            # Smile detection within the face region
            smiles = smile_cascade.detectMultiScale(face_region_gray, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(face_region_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def detect_face(request):
    return render(request, 'detect_face.html')

EMOTION_CATEGORIES = ["Happy", "Calm", "Angry", "Sad"]

def extract_features(audio_path):
    # audio_path = convert_to_wav(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, spectral_contrast])

def knn_fallback(features):
    templates = {
        "Happy": [37, 7, 16, 1.2, 0.5],
        "Angry": [33, 7, 14, 0.85, -1.5],
        "Calm":  [28, 4.5, 11, 0.95, 0.2],
        "Sad":   [22, 3.5, 9, 0.8, 0.3]
    }
    eps = 1e-6
    current_stats = [
        np.mean(features[:13]),
        np.std(features[:13]),
        np.ptp(features[:13]),
        np.mean(features[6:10])/(np.mean(features[:5]) + eps),
        np.mean(np.diff(features[:13]))
    ]
    distances = {}
    for emotion, template in templates.items():
        distances[emotion] = np.linalg.norm(np.array(current_stats) - np.array(template))
    return min(distances, key=distances.get)
def analyze_emotion(features):
    mfcc = features[:13]
    mean_mfcc = np.mean(mfcc)
    std_mfcc = np.std(mfcc)
    mfcc_range = np.ptp(mfcc)
    min_mfcc = np.min(features)
    max_mfcc = np.max(features)
    normalized_mfcc = (mean_mfcc - min_mfcc) / (max_mfcc - min_mfcc)
    eps = 1e-6
    high_freq_ratio = np.mean(mfcc[6:10]) / np.mean(mfcc[:5] + eps)
    spectral_flux = np.mean(np.diff(mfcc))
    if mean_mfcc > 35 and std_mfcc > 6 and high_freq_ratio > 1.1:
        return "Happy"
    elif 30 < mean_mfcc <= 35 and spectral_flux < -1 and mfcc_range > 12:
        return "Angry"
    elif 25 < mean_mfcc <= 30 and std_mfcc < 5 and spectral_flux > -0.5:
        return "Calm"
    elif mean_mfcc <= 25 and mfcc_range < 10 and high_freq_ratio < 0.9:
        return "Sad"
    else:
        return knn_fallback(features)
    # if normalized_mfcc > 0.6:
    #     return "Happy"
    # elif 0.50 < normalized_mfcc <= 0.6:
    #     return "Angry"
    # elif 0.40 < normalized_mfcc <= 0.50:
    #     return "Calm"
    # else:
    #     return "Sad"
from django.views.decorators.csrf import csrf_exempt
import os
from django.core.files.storage import FileSystemStorage
# @csrf_exempt
@csrf_exempt
def analyze_voice(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        file_path = "recorded_audio.wav"
        with open(file_path, "wb") as f:
            f.write(audio_file.read())
        # Now that the file is already in WAV format, you can process it
        y, sr = librosa.load(file_path, sr=None)
        # Example: just compute the duration
        duration = len(y) / sr
        os.remove(file_path)
        # For demonstration, we send the duration as emotion
        return JsonResponse({"emotion": f"Duration: {duration:.2f} seconds"})
    return JsonResponse({"error": "Invalid request"}, status=400)
def analyze_voice(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]
        file_path = f"recorded_audio.wav"
        with open(file_path, "wb") as f:
            f.write(audio_file.read())
        features = extract_features(file_path)
        emotion = analyze_emotion(features)
        os.remove(file_path)
        return JsonResponse({"emotion": emotion})
    return JsonResponse({"error": "Invalid request"}, status=400)

def detect_voice(request):
    return render(request,"detect_voice.html")
