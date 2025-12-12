
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import pytesseract
import json
import os
import logging

# Enable logging to see what's detected
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# === CONFIG ===
genai.configure(api_key="AIzaSyAdlsWQLDLGOvuzWsUq3CpvbRHwvaSSFKM")
gemini = genai.GenerativeModel('gemini-1.5-flash')

# === MODELS ===
model = YOLO("yolo11l.pt")  # This model HAS "book" class!
emotion_model = YOLO("emotion.onnx")
emotion_classes = ["Angry", "Fearful", "Happy", "Neutral", "Sad"]

# === CACHE ===
cache_file = "ar_cache.json"
cache = {}
if os.path.exists(cache_file):
    try:
        cache = json.load(open(cache_file))
    except:
        cache = {}

def detect_emotion(face_crop):
    if face_crop.size == 0 or min(face_crop.shape[:2]) < 48:
        return "Neutral"
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.merge([gray, gray, gray])
    try:
        res = emotion_model(gray3, conf=0.25, verbose=False)[0]
        if len(res.boxes) == 0:
            return "Neutral"
        emo = emotion_classes[int(res.boxes[0].cls[0])]
        conf = float(res.boxes[0].conf[0])
        if emo in ["Sad", "Fearful"] and conf < 0.65:
            return "Neutral"
        return emo
    except:
        return "Neutral"

def enhance_image_for_ocr(img):
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if min(h, w) < 200:
        scale = max(2, 300 // min(h, w))
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_OTSU)
    return thresh

@app.post("/frame")
async def frame(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Lower confidence to catch books!
    results = model(frame, conf=0.4, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        key = f"{label}_{cx}_{cy}"

        emotion = ""
        if label == "person":
            face = frame[y1:y2, x1:x2]
            emotion = detect_emotion(face)

        info = cache.get(key, "")

        detections.append({
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "emotion": emotion,
            "info": info
        })

    return {"detections": detections}

@app.post("/click")
async def click(file: UploadFile, x: int = Form(...), y: int = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"text": "Invalid image"}

    # Very low confidence to catch books
    results = model(frame, conf=0.25, verbose=False)[0]

    logging.info(f"Click at ({x}, {y})")

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        logging.info(f"Detected: {label} ({conf:.2f}) at ({x1},{y1})-({x2},{y2})")

        if x1 <= x <= x2 and y1 <= y <= y2:
            logging.info(f"CLICKED ON: {label}")

            cx = int((x1 + x2) // 2)
            cy = int((y1 + y2) // 2)
            key = f"{label}_{cx}_{cy}"

            if key in cache:
                return {"text": cache[key]}

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            enhanced = enhance_image_for_ocr(crop)
            text = pytesseract.image_to_string(enhanced, lang='eng', config='--psm 6')
            text = ' '.join(text.split())  # Clean up

            if not text or len(text) < 3:
                text = "[No text found]"

            try:
                response = gemini.generate_content(
                    f"Make a short fun caption about this book: \"{text[:200]}\""
                )
                caption = response.text.strip()
            except Exception as e:
                logging.error(f"Gemini error: {e}")
                caption = f"Book says: {text[:100]}"

            final_text = f"{caption}\n\n(OCR: {text[:120]}...)"
            cache[key] = final_text
            json.dump(cache, open(cache_file, "w"), indent=2)

            return {"text": final_text}

    return {"text": "No object clicked. Try clicking directly on a book!"}

print("AR Server Running → http://127.0.0.1:8000")