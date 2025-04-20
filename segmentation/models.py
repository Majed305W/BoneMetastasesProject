from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Patient(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='patient_profile')
    full_name = models.CharField(max_length=100)
    patient_id = models.CharField(max_length=50, unique=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=[('Male', 'Male'), ('Female', 'Female')], blank=True)
    contact = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.full_name} ({self.patient_id})"

class Study(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='studies')
    study_date = models.DateField()
    series_id = models.CharField(max_length=20)
    original_image = models.ImageField(upload_to='Patients/')
    result_image = models.ImageField(upload_to='Patients/')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Study {self.series_id} on {self.study_date} for {self.patient.full_name}"

class LabResult(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='lab_results')
    test_name = models.CharField(max_length=100)
    value = models.CharField(max_length=50)
    units = models.CharField(max_length=20, blank=True)
    reference_range = models.CharField(max_length=50, blank=True)
    date = models.DateField()

    def __str__(self):
        return f"{self.test_name} - {self.value} ({self.date})"

class TreatmentPlan(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='treatment_plans')
    plan_description = models.TextField()
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Plan for {self.patient.full_name} (updated {self.updated_at.date()})"
from django.db import models

# Create your models here.
