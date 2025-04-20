from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Home page
    path('delete_case/', views.delete_case, name='delete_case'),  # Delete route
    path('patients/', views.patient_list, name='patient_list'),  # 👈 New route for patient list
    path('patients/<str:patient_id>/', views.patient_detail, name='patient_detail'),
    path('patients/add/', views.add_patient, name='add_patient'),
    path('upload_report/', views.upload_report, name='upload_report'),
    path('upload_labs/', views.upload_labs, name='upload_labs'),
    path('patients/add/', views.add_patient, name='add_patient'),

]
