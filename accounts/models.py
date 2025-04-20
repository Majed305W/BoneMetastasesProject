from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    ROLE_CHOICES = [
        ('radiologist', 'Radiologist'),
        ('technician', 'Technician'),
        ('admin', 'Admin'),
        #('patient', 'Patient'),  # ✅ Add patient role
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='technician')

    def __str__(self):
        return f"{self.username} ({self.role})"
