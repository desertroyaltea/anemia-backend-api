from PIL import Image
import numpy as np
import cv2
import sys
import onnxruntime as ort # To run the ONNX model
from pathlib import Path

def detect_conjunctiva_yolo(pil_image: Image.Image, session: ort.InferenceSession, class_names: list):
    """
    Identifies the conjunctiva using a local YOLOv8 ONNX model.
    This version uses smarter logic to select the best detection.
    """
    model_input_size = (640, 640)
    original_size = pil_image.size
    
    # Pre-process the image
    image = pil_image.resize(model_input_size)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1)) # HWC to CHW
    image_np = np.expand_dims(image_np, axis=0)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: image_np})[0]

    # Post-process the output
    outputs = np.transpose(np.squeeze(outputs))
    
    detections = []
    x_factor = original_size[0] / model_input_size[0]
    y_factor = original_size[1] / model_input_size[1]

    for i in range(outputs.shape[0]):
        classes_scores = outputs[i][4:]
        class_id = np.argmax(classes_scores)
        max_score = np.amax(classes_scores)

        # Stricter confidence threshold
        if max_score >= 0.40:
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            detections.append({
                "box": [left, top, width, height],
                "score": max_score,
                "class_id": class_id
            })
    
    if not detections: 
        return None, None, None

    # --- NEW SMARTER SELECTION LOGIC ---
    # Prioritize 'forniceal_palpebral' (ID 1), then 'forniceal' (ID 0)
    detections.sort(key=lambda d: (
        d['class_id'] != 1, # 'forniceal_palpebral' comes first
        d['class_id'] != 0, # 'forniceal' comes second
        -d['score'] # Then sort by highest score
    ))
    
    best_detection = detections[0]
    best_box = best_detection["box"]
    best_class_name = class_names[best_detection["class_id"]]
    best_score = best_detection["score"]
    
    left, top, width, height = best_box
    
    # Crop the original image
    cropped_image = pil_image.crop((left, top, left + width, top + height))
    
    # Return everything for debugging
    return cropped_image, best_box, f"{best_class_name} ({best_score:.2f})"


if __name__ == "__main__":
    input_filename = "me.jpg" if len(sys.argv) <= 1 else sys.argv[1]
    output_filename = "cropped_output.jpg"
    debug_filename = "debug_labeled_output.jpg"
    model_path = Path("models") / "detector.onnx"
    
    # The class names from your training, in order
    CLASS_NAMES = ['forniceal', 'forniceal_palpebral', 'palpebral']

    # --- Load Models ---
    try:
        print(f"Loading object detection model from '{model_path}'...")
        detector_session = ort.InferenceSession(str(model_path))
    except Exception as e:
        print(f"❌ Error: Could not load the ONNX model. Make sure 'models/detector.onnx' exists.")
        print(f"   Details: {e}")
        sys.exit()

    try:
        print(f"Loading image '{input_filename}'...")
        image = Image.open(input_filename).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_filename}' was not found.")
        sys.exit()

    # --- Run Detection ---
    print("Attempting to detect and crop the conjunctiva...")
    cropped_image, bounding_box, label_text = detect_conjunctiva_yolo(image, detector_session, CLASS_NAMES)

    # --- Process Results ---
    if cropped_image and bounding_box:
        # Save and show the final cropped image
        print(f"✅ Success! Cropped image saved as '{output_filename}'.")
        cropped_image.save(output_filename)
        cropped_image.show()

        # Create and save the debug image with the label
        debug_image = np.array(image.copy())
        left, top, width, height = bounding_box
        
        # Draw the bounding box in bright green
        cv2.rectangle(debug_image, (left, top), (left + width, top + height), (0, 255, 0), 3)
        
        # --- NEW: Add the class name and score to the debug image ---
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(debug_image, (left, top - text_height - 10), (left + text_width, top), (0, 255, 0), -1)
        cv2.putText(debug_image, label_text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        print(f"💾 Saving debug image with label as '{debug_filename}'.")
        Image.fromarray(debug_image).save(debug_filename)

    else:
        print("❌ Failure: Could not detect a valid conjunctiva in the image.")

