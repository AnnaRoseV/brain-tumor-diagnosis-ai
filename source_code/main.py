from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import numpy as np
import onnxruntime as ort
import tflite_runtime.interpreter as tflite
import io
from PIL import Image
import cv2
import os
import uuid

# =====================================================
# APP SETUP
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_SIZE = 224
SEG_SIZE = 256

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve segmentation outputs
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# =====================================================
# LOAD MODELS
# =====================================================

# -------- Classification (TFLite) --------
cls_interpreter = tflite.Interpreter(
    model_path=os.path.join(BASE_DIR, "model_quant.tflite")
)
cls_interpreter.allocate_tensors()

cls_input_details = cls_interpreter.get_input_details()
cls_output_details = cls_interpreter.get_output_details()

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# -------- Segmentation (ONNX) --------
seg_session = ort.InferenceSession(
    os.path.join(BASE_DIR, "brisc_unet_segmentation.onnx")
)

seg_input_name = seg_session.get_inputs()[0].name
seg_output_name = seg_session.get_outputs()[0].name

# =====================================================
# SERVE FRONTEND
# =====================================================

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img_bytes = await file.read()
    img_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    original = np.array(img_rgb)

    # ---------------- Classification ----------------

    img_class = img_rgb.resize((CLASS_SIZE, CLASS_SIZE))
    class_array = np.array(img_class) / 255.0
    class_array = np.expand_dims(class_array, axis=0).astype(np.float32)

    cls_interpreter.set_tensor(
        cls_input_details[0]['index'], class_array
    )
    cls_interpreter.invoke()

    preds = cls_interpreter.get_tensor(
        cls_output_details[0]['index']
    )

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    tumor_type = CLASS_NAMES[idx]

    clean_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    percentage = 0.0
    level = "Normal"

    # ---------------- Segmentation ----------------

    if tumor_type != "notumor":

        seg_img = cv2.resize(original, (SEG_SIZE, SEG_SIZE))
        seg_img = seg_img.astype(np.float32) / 255.0
        seg_img = np.transpose(seg_img, (2, 0, 1))
        seg_img = np.expand_dims(seg_img, axis=0)

        seg_pred = seg_session.run(
            [seg_output_name],
            {seg_input_name: seg_img}
        )[0][0, 0]

        binary_mask = (seg_pred > 0.5).astype(np.uint8)

        clean_mask = cv2.resize(
            binary_mask,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        kernel = np.ones((5, 5), np.uint8)
        clean_mask = cv2.morphologyEx(
            clean_mask, cv2.MORPH_OPEN, kernel
        )

        tumor_pixels = np.sum(clean_mask)
        total_pixels = clean_mask.size
        percentage = (tumor_pixels / total_pixels) * 100

        if percentage < 5:
            level = "Mild"
        elif percentage < 20:
            level = "Moderate"
        else:
            level = "Severe"

    # ---------------- Save Images ----------------

    uid = str(uuid.uuid4())

    mask_path = os.path.join(OUTPUT_DIR, f"mask_{uid}.png")
    overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{uid}.png")

    cv2.imwrite(mask_path, (clean_mask * 255).astype(np.uint8))

    overlay = original.copy()
    overlay[clean_mask == 1] = [255, 0, 0]

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    # ---------------- Response ----------------

    return {
        "tumor_type": tumor_type,
        "confidence": round(confidence, 4),
        "tumor_percentage": round(percentage, 2),
        "tumor_level": level,
        "mask_image": f"/outputs/{os.path.basename(mask_path)}",
        "overlay_image": f"/outputs/{os.path.basename(overlay_path)}"
    }