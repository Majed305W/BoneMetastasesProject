from django.contrib import admin
from .models import Patient, Study, LabResult, TreatmentPlan

# Register your models here
admin.site.register(Patient)
admin.site.register(Study)
admin.site.register(LabResult)
admin.site.register(TreatmentPlan)
