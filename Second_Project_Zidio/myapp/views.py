from django.shortcuts import render
from django.http import JsonResponse
import soundfile as sf
import librosa
import os
import joblib
import ffmpeg
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
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
def analyze_eyebrows(gray_face, eyes):
    if len(eyes) == 0:
        return "unknown"
    eye_y = min([ey for (ex, ey, ew, eh) in eyes])
    forehead_region = gray_face[:eye_y, :]
    edges = cv2.Canny(forehead_region, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 10]
        if len(horizontal_lines) > 1:  
            return "furrowed"
    return "normal"

def analyze_mouth(gray_face, smiles):
    if len(smiles) > 0:
        for (sx, sy, sw, sh) in smiles:
            mouth_region = gray_face[sy:sy + sh, sx:sx + sw]
            top_half = mouth_region[:sh//2, :]
            bottom_half = mouth_region[sh//2:, :]
            top_edges = np.sum(cv2.Canny(top_half, 50, 150))
            bottom_edges = np.sum(cv2.Canny(bottom_half, 50, 150))
            if top_edges > bottom_edges:
                return "upward"  # Happy
            else:
                return "downward"  # Sad
    return "neutral"
# Emotion classification based on face brightness, smile, and eye detection
def classify_emotion(gray_face, face_region_color):
    global recent_emotions
    mean_intensity = np.mean(gray_face)
    smiles = smile_cascade.detectMultiScale(
        gray_face, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25)
    )
    eyes = eye_cascade.detectMultiScale(
        gray_face, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20)
    )
    # smiles = smile_cascade.detectMultiScale(gray_face, scaleFactor=1.8, minNeighbors=25, minSize=(25, 25))
    # eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=15, minSize=(20, 20))
    eyebrow_status = analyze_eyebrows(gray_face, eyes)
    mouth_status = analyze_mouth(gray_face, smiles)
    score=3
    # Decision rules 
    if len(smiles) > 0 and mouth_status == "upward" and eyebrow_status=="normal":
        # emotion = "Happy (Normal)"
        score +=3
    # elif len(eyes) == 0 or (mouth_status=="downward" and mean_intensity<100): 
        # emotion = "Suicidal"
    elif eyebrow_status == "furrowed":
        score -= 2
    if mouth_status == "downward":
        if mean_intensity < 100:
            score -= 3
        else:
            score -= 1
    if mean_intensity < 80:
        score -= 2
    elif mean_intensity > 150:
        score += 2
    if len(eyes) == 0:
        score -= 3
    if score >= 3:
        emotion = "Happy (Normal)"
    elif score <= -3:
        emotion = "Suicidal"
    elif score < 0 and score >-3:
        emotion = "Anxiety"
    else:
        emotion = "Depression"
    # elif mean_intensity > 130 or eyebrow_status=="furrowed":
    #     emotion = "Anxiety"
    # elif 130 < mean_intensity <= 180:
    #     emotion = "Anxiety"
    # elif mouth_status == "downward" or (80 < mean_intensity <= 130):
    #     emotion = "Depression"
    # else:
    #     emotion = "Normal"

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

# def convert_to_wav(input_path, output_path):
#     try:
#         ffmpeg.input(input_path).output(
#             output_path,
#             format = 'wav',
#             acodec = 'pcm_s16le',
#             ac=1,
#             ar='22050'
#         ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
#         return True
#     except Exception as e:
#         print("Error during conversion:",e)
#         return True 
import subprocess
def convert_to_wav(input_path, output_path):
    """
    Convert the input audio file to a WAV file using the ffmpeg command-line tool.
    The output is in WAV format with PCM S16LE codec, mono channel, and 22050 Hz sampling rate.
    """
    try:
        # Call ffmpeg using subprocess.
        # The '-y' flag overwrites the output file if it exists.
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '22050', output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print("Error during conversion:", e)
        return False

    
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)  
    mfcc_std = np.std(mfcc, axis=1)
    # Compute pitch using pyin
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0  # Handle all NaN case
    pitch_std = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0
    # Compute RMS energy
    rms = librosa.feature.rms(y=y)
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    voiced_frames = np.sum(voiced_flag)
    duration = len(y) / sr
    speaking_rate = voiced_frames / duration if duration > 0 else 0
    
    # Concatenate all features into a single vector
    features = np.concatenate([
        mfcc_mean,          # 20 elements
        mfcc_std,           # 20 elements
        [pitch_mean, pitch_std, energy_mean, energy_std, speaking_rate]  # 5 elements
    ])
    return features
    

def detect_emotion(features):
    pitch_mean = features[40]    
    energy_mean = features[42]   
    speaking_rate = features[44] 
    if pitch_mean > 180 and energy_mean > 0.1 and speaking_rate > 10:
        return "Happy"
    elif pitch_mean < 120 and energy_mean < 0.05 and speaking_rate < 5:
        return "Depressed"
    else:
        return "Normal"
@csrf_exempt  # Use with caution; better to handle CSRF properly in production.
def analyze_voice(request):
    """
    Handle the audio file upload, conversion, feature extraction, and classification of voice emotion.
    """
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        input_path = 'temp_input_audio'
        output_path = 'temp_converted_audio.wav'
        
        # Save the uploaded file locally.
        with open(input_path, 'wb') as f:
            f.write(audio_file.read())
        
        # Convert the input audio file to a proper WAV format.
        conversion_success = convert_to_wav(input_path, output_path)
        if not conversion_success:
            os.remove(input_path)
            return JsonResponse({"error": "Audio conversion failed."}, status=500)
        
        # Extract audio features using librosa.
        features = extract_features(output_path)
        emotion = detect_emotion(features)
        
        # Clean up temporary files.
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return JsonResponse({"emotion": emotion})
    else:
        return JsonResponse({"error": "Invalid request. Please upload an audio file."}, status=400)

def detect_voice(request):
    return render(request, 'detect_voice.html')
