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
import onnxruntime as ort

class ImageData(BaseModel):
    image_b64: str

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

HB_ESTIMATION_RUN_DIR = Path("models") / "run_20250918_192217"
HB_FEATURES = ["glare_frac", "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50", "gray_p90", "gray_kurt", "gray_std", "gray_mean", "B_mean", "B_p10", "B_p75", "G_kurt", "mean_vesselness", "p90_vesselness", "skeleton_len_per_area", "branchpoint_density", "tortuosity_mean", "vessel_area_fraction"]

try:
    hb_model_path = HB_ESTIMATION_RUN_DIR / "hb_rf.joblib"
    HB_MODEL = joblib.load(hb_model_path)
    print("Hemoglobin model loaded successfully.")

    detector_path = Path("models") / "detector.onnx"
    DETECTOR_SESSION = ort.InferenceSession(str(detector_path))
    print("Object detector model loaded successfully.")
except Exception as e:
    HB_MODEL, DETECTOR_SESSION = None, None
    print(f"CRITICAL: Could not load models. Error: {e}")

def detect_conjunctiva_local(pil_image: Image.Image):
    model_input_size = (640, 640)
    original_size = pil_image.size
    image = pil_image.resize(model_input_size); image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1)); image_np = np.expand_dims(image_np, axis=0)
    input_name = DETECTOR_SESSION.get_inputs()[0].name; output_name = DETECTOR_SESSION.get_outputs()[0].name
    outputs = DETECTOR_SESSION.run([output_name], {input_name: image_np})[0]
    outputs = np.transpose(np.squeeze(outputs)); boxes, scores = [], []
    x_factor, y_factor = original_size[0] / model_input_size[0], original_size[1] / model_input_size[1]
    for i in range(outputs.shape[0]):
        classes_scores = outputs[i][4:]; max_score = np.amax(classes_scores)
        if max_score >= 0.25:
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor); top = int((y - h / 2) * y_factor)
            width = int(w * x_factor); height = int(h * y_factor)
            scores.append(max_score); boxes.append([left, top, width, height])
    if not boxes: return None
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    if len(indices) > 0:
        best_box_index = indices.flatten()[0]; best_box = boxes[best_box_index]
        left, top, width, height = best_box
        return pil_image.crop((left, top, left + width, top + height))
    else: return None

def kurtosis_numpy(data): mean = np.mean(data); std_dev = np.std(data); return np.mean(((data - mean) / std_dev) ** 4) if std_dev > 0 else 0
def detect_glare_mask(rgb: np.ndarray) -> np.ndarray: hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32); S, V = hsv[..., 1] / 255.0, hsv[..., 2] / 255.0; mask_hsv = (V > 0.90) & (S < 0.25); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0; hi = float(np.quantile(gray, 0.995)); mask_gray = gray >= hi; mask = cv2.morphologyEx((mask_hsv | mask_gray).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1); return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
def inpaint_glare(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray: bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR); out = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), inpaintRadius=3, flags=cv2.INPAINT_TELEA); return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
def compute_baseline_features(pil_img: Image.Image) -> dict: rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8); R, G, B = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32); S = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0; a = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[..., 1].astype(np.float32) - 128.0; R_norm = R / (R + G + B + 1e-6); return {"R_p50": np.percentile(R, 50), "R_norm_p50": np.percentile(R_norm, 50), "a_mean": np.mean(a), "R_p10": np.percentile(R, 10), "gray_mean": np.mean(gray), "RG": np.mean(R) / (np.mean(G) + 1e-6), "gray_kurt": kurtosis_numpy(gray.ravel()), "gray_p90": np.percentile(gray, 90), "S_p50": np.percentile(S, 50), "B_p10": np.percentile(B, 10), "B_mean": np.mean(B), "gray_std": np.std(gray), "B_p75": np.percentile(B, 75), "G_kurt": kurtosis_numpy(G.ravel())}
def vascularity_features_from_conjunctiva(rgb_u8: np.ndarray) -> dict:
    g = rgb_u8[..., 1].astype(np.uint8)
    g_eq = exposure.equalize_adapthist(g, clip_limit=0.01)
    vmap = filters.frangi(g_eq, sigmas=np.arange(1, 6, 1), alpha=0.5, beta=0.5, black_ridges=True)
    vmap = (vmap - vmap.min()) / (np.ptp(vmap) + 1e-8)
    mask = vmap > filters.threshold_otsu(vmap)
    mask = morphology.remove_small_objects(mask, min_size=50)
    mask = morphology.remove_small_holes(mask, area_threshold=50)
    skel = skeletonize(mask)
    area = float(mask.shape[0] * mask.shape[1])
    neigh = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT)
    branches = ((skel) & (neigh >= 4))
    lbl = measure.label(skel, connectivity=2)
    torts = []
    for region in measure.regionprops(lbl):
        coords = np.array(region.coords)
        if coords.shape[0] < 10: continue
        chord = np.linalg.norm(coords.max(0) - coords.min(0)) + 1e-8
        torts.append(float(coords.shape[0]) / chord)
    return {"vessel_area_fraction": mask.sum() / area, "mean_vesselness": vmap.mean(), "p90_vesselness": np.percentile(vmap, 90), "skeleton_len_per_area": skel.sum() / area, "branchpoint_density": branches.sum() / area, "tortuosity_mean": np.mean(torts) if torts else 1.0}

@app.post("/api/analyze")
async def analyze_image(image_data: ImageData):
    if HB_MODEL is None or DETECTOR_SESSION is None: raise HTTPException(status_code=500, detail="A model is not loaded on the server.")
    try:
        image_bytes = base64.b64decode(image_data.image_b64)
        full_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        crop_image = detect_conjunctiva_local(full_image)
        if crop_image is None: raise HTTPException(status_code=400, detail="Could not detect conjunctiva in the image.")
        rgb = np.array(crop_image.convert("RGB"), dtype=np.uint8)
        glare_mask = detect_glare_mask(rgb); rgb_proc = inpaint_glare(rgb, glare_mask) if glare_mask.sum() > 0 else rgb
        feats = {"glare_frac": float(glare_mask.mean())}; feats.update(compute_baseline_features(Image.fromarray(rgb_proc))); feats.update(vascularity_features_from_conjunctiva(rgb_proc))
        x_vec = np.array([[feats.get(f, 0.0) for f in HB_FEATURES]], dtype=np.float32)
        hb_pred = float(HB_MODEL.predict(x_vec)[0])
        buffered = io.BytesIO(); crop_image.save(buffered, format="JPEG")
        crop_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {"hb_value": hb_pred, "crop_b64": crop_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")