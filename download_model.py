from roboflow import Roboflow
import os

ROBOFLOW_API_KEY = "jMhyBQxeQvj69nttV0mN"  # Your API Key
MODEL_ID = "eye-conjunctiva-detector/2"

def download_tf_model():
    if "YOUR_PRIVATE_API_KEY" in ROBOFLOW_API_KEY:
        print("Please paste your Roboflow Private API Key into the script.")
        return

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    project_id, version_number = MODEL_ID.split('/')
    project = rf.project(project_id)
    version = project.version(int(version_number))

    # Using the "tensorflow" format, as specified in the error message
    print(f"Downloading TensorFlow Lite model for '{MODEL_ID}'...")
    version.download("tensorflow")

    print("\n✅ Download complete!")
    print("Look for a new folder in this directory containing your model.")
    print("Inside, you should find a '.tflite' file.")

if __name__ == "__main__":
    download_tf_model()