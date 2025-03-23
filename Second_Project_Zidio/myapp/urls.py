from django.urls import path
from .views import home, detect_mood, suggest_task, detect_face, video_feed
from .views import analyze_voice, detect_voice, detect_emotion
# from .views import predict
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', home, name='home'),
    path('detect-mood/', detect_mood, name='detect_mood'),
    path('suggest-task/<str:mood>/', suggest_task, name='suggest_task'),
    path('detect_face/', detect_face, name='detect_face'),
    path('video_feed/', video_feed, name='video_feed'),
    path("detect_voice/", detect_voice, name="detect_voice"),
    path("analyze_voice/", analyze_voice, name="analyze_voice"),
    # path('predict_audio/', predict, name='predict_audio')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
