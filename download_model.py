from roboflow import Roboflow
import os

ROBOFLOW_API_KEY = "jMhyBQxeQvj69nttV0mN"
MODEL_ID = "eye-conjunctiva-detector/2"

def download_darknet_model():
    if ROBOFLOW_API_KEY == "YOUR_PRIVATE_API_KEY":
        print("Please paste your Roboflow Private API Key into the script.")
        return

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    project_id, version_number = MODEL_ID.split('/')
    project = rf.project(project_id)
    version = project.version(int(version_number))

    print(f"Downloading Darknet model for '{MODEL_ID}'...")
    # Change is here: "onnx" is now "darknet"
    version.download("darknet")

    print("\n✅ Download complete!")
    print(f"Look for a folder named '{project_id}.{version_number}' in this directory.")
    print("Inside that folder, you will find '.weights' and '.cfg' model files.")

if __name__ == "__main__":
    download_darknet_model()