import requests

ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# Step 1: Get all instances (DICOM images)
instances = requests.get(f"{ORTHANC_URL}/instances", auth=AUTH).json()

# Step 2: Pick the first image (you can loop if needed)
instance_id = instances[0]
print(f"Found instance ID: {instance_id}")

# Step 3: Download the DICOM file
dicom_data = requests.get(f"{ORTHANC_URL}/instances/{instance_id}/file", auth=AUTH)

# Step 4: Save the file locally
with open("downloaded_image.dcm", "wb") as f:
    f.write(dicom_data.content)

print("DICOM downloaded as downloaded_image.dcm ✅")
