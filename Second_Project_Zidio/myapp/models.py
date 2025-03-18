from django.db import models

# Create your models here.
from django.utils.timezone import now

class MoodEntry(models.Model):
    text = models.TextField()  # Store the answers given by the user
    mood = models.CharField(max_length=50)  # Detected mood category
    timestamp = models.DateTimeField(default=now)  # Auto timestamp

    def __str__(self):
        return f"{self.mood} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
