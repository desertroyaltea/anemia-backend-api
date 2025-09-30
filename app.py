from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from PIL import Image
import numpy as np
import base64
import io
import cv2
from skimage import exposure, filters, morphology, measure
from skimage.morphology import skeletonize
from pathlib import Path

# --- Pydantic model for request body ---
class ImageData(BaseModel):
    image_b64: str

# --- Initialize FastAPI App ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- CONFIG & CONSTANTS ---
HB_ESTIMATION_RUN_DIR = Path("models") / "run_20250-09-18_192217"
HB_FEATURES = ["glare_frac", "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50", "gray_p90", "gray_kurt", "gray_std", "gray_mean", "B_mean", "B_p10", "B_p75", "G_kurt", "mean_vesselness", "p90_vesselness", "skeleton_len_per_area", "branchpoint_density", "tortuosity_mean", "vessel_area_fraction"]

# --- MODEL LOADING ---
try:
    hb_model_path = HB_ESTIMATION_RUN_DIR / "hb_rf.joblib"
    HB_MODEL = joblib.load(hb_model_path)
    print("Hemoglobin model loaded successfully.")

    weights_path = str(Path("models") / "yolov4.weights")
    cfg_path = str(Path("models") / "yolov4.cfg")
    DETECTOR_NET = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    print("Object detector model loaded successfully.")
    
except Exception as e:
    HB_MODEL, DETECTOR_NET = None, None
    print(f"CRITICAL: Could not load models. Error: {e}")

# --- NEW: LOCAL OBJECT DETECTION FUNCTION (using OpenCV DNN) ---
def detect_conjunctiva_local(pil_image: Image.Image):
    np_image = np.array(pil_image)
    H, W, _ = np_image.shape

    # Pre-processing for Darknet model
    blob = cv2.dnn.blobFromImage(np_image, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Run inference
    DETECTOR_NET.setInput(blob)
    layer_names = DETECTOR_NET.getLayerNames()
    output_layers = [layer_names[i - 1] for i in DETECTOR_NET.getUnconnectedOutLayers()]
    outputs = DETECTOR_NET.forward(output_layers)

    # Post-processing to find the best bounding box
    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25: # Confidence threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    if not boxes: return None

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    if len(indices) == 0: return None
    
    best_box_index = indices[0]
    x, y, w, h = boxes[best_box_index]
    return pil_image.crop((x, y, x + w, y + h))

# --- FEATURE EXTRACTION & OTHER HELPERS (Unchanged) ---
def kurtosis_numpy(data): mean = np.mean(data); std_dev = np.std(data); return np.mean(((data - mean) / std_dev) ** 4) if std_dev > 0 else 0
def detect_glare_mask(rgb: np.ndarray) -> np.ndarray: hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32); S, V = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0; mask_hsv = (V > 0.90) & (S < 0.25); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0; hi = float(np.quantile(gray, 0.995)); mask_gray = gray >= hi; mask = cv2.morphologyEx((mask_hsv | mask_gray).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1); return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
def inpaint_glare(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray: bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); out = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), inpaintRadius=3, flags=cv2.INPAINT_TELEA); return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
def compute_baseline_features(pil_img: Image.Image) -> dict: rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8); R, G, B = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32); S = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0; a = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., 1].astype(np.float32) - 128.0; R_norm = R / (R + G + B + 1e-6); return {"R_p50": np.percentile(R, 50), "R_norm_p50": np.percentile(R_norm, 50), "a_mean": np.mean(a), "R_p10": np.percentile(R, 10), "gray_mean": np.mean(gray), "RG": np.mean(R) / (np.mean(G) + 1e-6), "gray_kurt": kurtosis_numpy(gray.ravel()), "gray_p90": np.percentile(gray, 90), "S_p50": np.percentile(S, 50), "B_p10": np.percentile(B, 10), "B_mean": np.mean(B), "gray_std": np.std(gray), "B_p75": np.percentile(B, 75), "G_kurt": kurtosis_numpy(G.ravel())}
def vascularity_features_from_conjunctiva(rgb_u8: np.ndarray) -> dict: g = rgb_u8[..., 1].astype(np.uint8); g_eq = exposure.equalize_adapthist(g, clip_limit=0.01); vmap = filters.frangi(g_eq, sigmas=np.arange(1, 6, 1), alpha=0.5, beta=0.5, black_ridges=True); vmap = (vmap - vmap.min()) / (np.ptp(vmap) + 1e-8); mask = vmap > filters.threshold_otsu(vmap); mask = morphology.remove_small_objects(mask, min_size=50); mask = morphology.remove_small_holes(mask, area_threshold=50); skel = skeletonize(mask); area = float(mask.shape[0] * mask.shape[1]); neigh = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT); branches = ((skel) & (neigh >= 4)); lbl, torts = measure.label(skel, connectivity=2), []; 
    for region in measure.regionprops(lbl): coords = np.array(region.coords); 
        if coords.shape[0] < 10: continue; chord = np.linalg.norm(coords.max(0) - coords.min(0)) + 1e-8; torts.append(float(coords.shape[0]) / chord); 
    return {"vessel_area_fraction": mask.sum() / area, "mean_vesselness": vmap.mean(), "p90_vesselness": np.percentile(vmap, 90), "skeleton_len_per_area": skel.sum() / area, "branchpoint_density": branches.sum() / area, "tortuosity_mean": np.mean(torts) if torts else 1.0}

# --- API ENDPOINT (Now uses local detection) ---
@app.post("/api/analyze")
async def analyze_image(image_data: ImageData):
    if HB_MODEL is None or DETECTOR_NET is None:
        raise HTTPException(status_code=500, detail="A model is not loaded on the server.")
    try:
        image_bytes = base64.b64decode(image_data.image_b64)
        full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        crop_image = detect_conjunctiva_local(full_image)
        if crop_image is None:
            raise HTTPException(status_code=400, detail="Could not detect conjunctiva in the image.")

        rgb = np.array(crop_image.convert("RGB"), dtype=np.uint8)
        glare_mask = detect_glare_mask(rgb); rgb_proc = inpaint_glare(rgb, glare_mask) if glare_mask.sum() > 0 else rgb
        feats = {"glare_frac": float(glare_mask.mean())}
        feats.update(compute_baseline_features(Image.fromarray(rgb_proc)))
        feats.update(vascularity_features_from_conjunctiva(rgb_proc))
        x_vec = np.array([[feats.get(f, 0.0) for f in HB_FEATURES]], dtype=np.float32)
        hb_pred = float(HB_MODEL.predict(x_vec)[0])
        
        buffered = io.BytesIO(); crop_image.save(buffered, format="JPEG")
        crop_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {"hb_value": hb_pred, "crop_b64": crop_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")