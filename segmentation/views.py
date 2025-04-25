import uuid
import csv
import os
import torch
import numpy as np
import cv2
import pydicom
from PIL import Image
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from torchvision import transforms
from .unet import Unet
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model once
model = Unet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load('model/Model_50.pth', map_location=device))
model.to(device)
model.eval()

# Helper function to check for DICOM
def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False

@login_required(login_url='/accounts/login/')
def index(request):
    uploaded_image = None
    result_image = None
    patient_name = ""
    patient_id = ""
    age = ""
    history = []

    patients_dir = os.path.join(settings.MEDIA_ROOT, 'Patients')
    os.makedirs(patients_dir, exist_ok=True)

    # Count total patients and studies
    total_patients = len(os.listdir(patients_dir))
    total_studies = sum(
        len(os.listdir(os.path.join(patients_dir, pid)))
        for pid in os.listdir(patients_dir)
    )

    if os.path.exists(patients_dir):
        for patient_folder in os.listdir(patients_dir):
            patient_path = os.path.join(patients_dir, patient_folder)
            for study_folder in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_folder)
                for series_folder in os.listdir(study_path):
                    series_path = os.path.join(study_path, series_folder)
                    original = os.path.join(series_path, "original.png")
                    result = os.path.join(series_path, "result.png")
                    if os.path.exists(original) and os.path.exists(result):
                        history.append((
                            os.path.join('media', 'Patients', patient_folder, study_folder, series_folder, 'original.png'),
                            os.path.join('media', 'Patients', patient_folder, study_folder, series_folder, 'result.png'),
                            patient_folder,
                            study_folder
                        ))

    if request.method == 'POST' and request.FILES.get('image'):
        patient_name = request.POST.get('patient_name', '')
        patient_id = request.POST.get('patient_id', '')
        age = request.POST.get('age', '')
        study_date = uuid.uuid4().hex[:8]
        series_id = uuid.uuid4().hex[:6]

        series_path = os.path.join(patients_dir, patient_id, study_date, series_id)
        os.makedirs(series_path, exist_ok=True)

        image_file = request.FILES['image']
        input_name = os.path.join(series_path, "original.png")
        result_name = os.path.join(series_path, "result.png")

        fs = FileSystemStorage()
        with open(input_name, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        if is_dicom_file(input_name):
            ds = pydicom.dcmread(input_name)
            pixel_array = ds.pixel_array
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            pixel_array = (pixel_array * 255).astype(np.uint8)
            image = Image.fromarray(pixel_array)
        else:
            image = Image.open(input_name)

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        output = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (output > 0.2).astype(np.uint8)
        mask_cleaned = cv2.medianBlur(mask, 5)

        cv2.imwrite(result_name, mask_cleaned * 255)

        uploaded_image = os.path.join('media', 'Patients', patient_id, study_date, series_id, 'original.png')
        result_image = os.path.join('media', 'Patients', patient_id, study_date, series_id, 'result.png')

        history.insert(0, (uploaded_image, result_image, patient_id, study_date))

        log_file = os.path.join(settings.MEDIA_ROOT, 'case_log.csv')
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['Patient Name', 'Patient ID', 'Age', 'Input Image', 'Result Image']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'Patient Name': patient_name,
                'Patient ID': patient_id,
                'Age': age,
                'Input Image': input_name,
                'Result Image': result_name
            })

    case_log = []
    log_file = os.path.join(settings.MEDIA_ROOT, 'case_log.csv')
    if os.path.isfile(log_file):
        with open(log_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                case_log.append(row)

    return render(request, 'index.html', {
        'uploaded_image': uploaded_image,
        'result_image': result_image,
        'patient_name': patient_name,
        'patient_id': patient_id,
        'age': age,
        'history': history,
        'case_log': case_log,
        'total_patients': total_patients,
        'total_studies': total_studies,
    })

@csrf_exempt
@login_required
def upload_report(request):
    if request.method == 'POST' and request.FILES.get('report_file'):
        report = request.FILES['report_file']
        save_path = os.path.join(settings.MEDIA_ROOT, 'Reports')
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, report.name), 'wb+') as f:
            for chunk in report.chunks():
                f.write(chunk)
    return redirect('index')

@csrf_exempt
@login_required
def upload_labs(request):
    if request.method == 'POST' and request.FILES.get('lab_file'):
        lab_file = request.FILES['lab_file']
        save_path = os.path.join(settings.MEDIA_ROOT, 'LabResults')
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, lab_file.name), 'wb+') as f:
            for chunk in lab_file.chunks():
                f.write(chunk)
    return redirect('index')

@csrf_exempt
def delete_case(request):
    if request.method == 'POST':
        input_path = request.POST.get('input_path', '')
        result_path = request.POST.get('result_path', '')

        base_dir = settings.MEDIA_ROOT
        input_file = os.path.join(base_dir, input_path.replace('/media/', ''))
        result_file = os.path.join(base_dir, result_path.replace('/media/', ''))

        try:
            if os.path.exists(input_file):
                os.remove(input_file)
            if os.path.exists(result_file):
                os.remove(result_file)
        except Exception as e:
            print("Error deleting files:", e)

    return redirect('/')

@login_required(login_url='/accounts/login/')
def patient_list(request):
    patients_dir = os.path.join(settings.MEDIA_ROOT, 'Patients')
    patients = []

    if os.path.exists(patients_dir):
        all_patients = sorted(os.listdir(patients_dir))

        query = request.GET.get('search', '')
        if query:
            patients = [pid for pid in all_patients if query.lower() in pid.lower()]
        else:
            patients = all_patients

    return render(request, 'patient_list.html', {
        'patients': patients,
        'search_query': request.GET.get('search', ''),
    })

@login_required(login_url='/accounts/login/')
def patient_detail(request, patient_id):
    patients_dir = os.path.join(settings.MEDIA_ROOT, 'Patients', patient_id)
    studies = []

    os.makedirs(patients_dir, exist_ok=True)

    info = {'Name': '', 'Age': '', 'Notes': ''}
    info_path = os.path.join(settings.MEDIA_ROOT, 'Patients', patient_id, 'info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    info[key.strip()] = value.strip()

    if request.method == 'POST' and request.FILES.get('image'):
        study_date = uuid.uuid4().hex[:8]
        series_id = uuid.uuid4().hex[:6]
        series_path = os.path.join(patients_dir, study_date, series_id)
        os.makedirs(series_path, exist_ok=True)

        image_file = request.FILES['image']
        input_name = os.path.join(series_path, "original.png")
        result_name = os.path.join(series_path, "result.png")

        with open(input_name, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        if is_dicom_file(input_name):
            ds = pydicom.dcmread(input_name)
            pixel_array = ds.pixel_array
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            pixel_array = (pixel_array * 255).astype(np.uint8)
            image = Image.fromarray(pixel_array)
        else:
            image = Image.open(input_name)

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)

        output = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (output > 0.2).astype(np.uint8)
        mask_cleaned = cv2.medianBlur(mask, 5)

        cv2.imwrite(result_name, mask_cleaned * 255)

    if os.path.exists(patients_dir):
        for study_folder in sorted(os.listdir(patients_dir), reverse=True):
            study_path = os.path.join(patients_dir, study_folder)
            for series_folder in os.listdir(study_path):
                series_path = os.path.join(study_path, series_folder)
                original = os.path.join(series_path, 'original.png')
                result = os.path.join(series_path, 'result.png')
                studies.append({
                    'study_date': study_folder,
                    'series_id': series_folder,
                    'original': os.path.join('media', 'Patients', patient_id, study_folder, series_folder, 'original.png') if os.path.exists(original) else None,
                    'result': os.path.join('media', 'Patients', patient_id, study_folder, series_folder, 'result.png') if os.path.exists(result) else None,
                })

    return render(request, 'patient_detail.html', {
        'patient_id': patient_id,
        'studies': studies,
        'info': info,
    })

@login_required(login_url='/accounts/login/')
def add_patient(request):
    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        patient_name = request.POST.get('patient_name')
        age = request.POST.get('age')
        notes = request.POST.get('notes', '')

        patient_folder = os.path.join(settings.MEDIA_ROOT, 'Patients', patient_id)
        os.makedirs(patient_folder, exist_ok=True)

        info_path = os.path.join(patient_folder, 'info.txt')
        with open(info_path, 'w') as f:
            f.write(f'Name: {patient_name}\n')
            f.write(f'Age: {age}\n')
            f.write(f'Notes: {notes}\n')

        return redirect('patient_detail', patient_id=patient_id)

    return render(request, 'add_patient.html')

# ✅ NEW VIEW TO SHOW STATIC DICOM & SEGMENTATION RESULTS
@login_required(login_url='/accounts/login/')
def dicom_result_view(request):
    return render(request, 'dicom_results.html', {
        'original_image': 'media/results/original_image.png',
        'segmentation_image': 'media/results/segmentation_result.png'
    })
