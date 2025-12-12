import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import pytesseract
import json
import os

# === CONFIG ===
GEMINI_API_KEY = "AIzaSyAdlsWQLDLGOvuzWsUq3CpvbRHwvaSSFKM"
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel('gemini-2.5-flash')

# === MODELS ===
model = YOLO("yolo11l.pt")
emotion_model = YOLO("emotion.onnx")
emotion_classes = ["Angry", "Fearful", "Happy", "Neutral", "Sad"]

# === CACHE & STATE ===
cache_file = "ar_cache.json"
cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
snapshot_file = "clicked_book.jpg"
current_results = None
last_ocr_printed = set()  # Prevent spam printing of same text

# === EMOTION DETECTION ===
def detect_emotion(face_crop):
    if face_crop.size == 0 or min(face_crop.shape[:2]) < 48:
        return "Neutral"
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    img_3ch = cv2.merge([gray, gray, gray])
    results = emotion_model(img_3ch, conf=0.25, verbose=False)
    if len(results[0].boxes) == 0:
        return "Neutral"
    box = results[0].boxes[0]
    emotion = emotion_classes[int(box.cls[0])]
    conf = float(box.conf[0])
    if emotion in ["Sad", "Fearful"] and conf < 0.65:
        return "Neutral"
    return emotion

# === MOUSE CALLBACK - OCR + PRINT + OVERLAY ===
def mouse_callback(event, x, y, flags, param):
    global current_results, display, frame_mirrored

    if event == cv2.EVENT_LBUTTONDOWN and current_results is not None:
        for box in current_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            obj_id = f"{label}_{x1}_{y1}"

            if x1 <= x <= x2 and y1 <= y <= y2:
                # Visual feedback
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 4)
                cv2.putText(display, "Reading text...", (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Crop and fix mirror
                crop = frame_mirrored[y1:y2, x1:x2]
                crop_ocr = cv2.flip(crop, 1)

                if crop_ocr.size > 10000:
                    cv2.imwrite(snapshot_file, crop_ocr)
                    detected_text = pytesseract.image_to_string(crop_ocr, lang='eng').strip()

                    # Print in terminal ONCE
                    if detected_text and detected_text not in last_ocr_printed:
                        print(f"\n[OCR] From {label}:")
                        print(detected_text)
                        last_ocr_printed.add(detected_text)

                    # Generate fun caption
                    try:
                        info = gemini.generate_content(
                            f"Make a short fun caption about a book/page saying: '{detected_text[:200]}'"
                        ).text.strip()[:100]
                    except:
                        info = detected_text[:80] + "..." if detected_text else "Text found!"

                    cache[obj_id] = info
                    json.dump(cache, open(cache_file, "w"))
                else:
                    info = "No text"

                # Overlay result
                cv2.putText(display, info, (x1, y2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                break

# === LEGEND ===
print("• Normal face → NEUTRAL")
print("• Big smile → HAPPY")
print("• Real sad → SAD\n")

# === MAIN LOOP ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("AR Book Reader", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("AR Book Reader", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_mirrored = cv2.flip(frame, 1)
    display = frame_mirrored.copy()

    results = model(frame_mirrored, conf=0.6, imgsz=640, verbose=False)[0]
    current_results = results

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        obj_id = f"{label}_{x1}_{y1}"

        color = (255, 100, 0) if label == "person" else (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if label == "person":
            face = frame_mirrored[y1:y2, x1:x2]
            emotion = detect_emotion(face)
            cv2.putText(display, emotion.upper(), (x1, y1-35),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 215, 255), 3)

        # Show OCR result or "Click me!"
        if obj_id in cache:
            cv2.putText(display, cache[obj_id], (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(display, "Click me!", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    # Instructions
    cv2.putText(display, "Click book → Read text | q = quit",
                (10, display.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Shiva_AR", display)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("AR app is closed")