import cv2
from ultralytics import YOLO
import os
import torch
import numpy as np
import tempfile # Included for completeness, though mainly used in app.py

# --- CONFIGURATION & MODEL INITIALIZATION ---

# Models folder should live inside the project directory next to this file
# e.g. <project>/models/yolov9c.pt
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Debug: list model directory contents to help diagnose missing files
if not os.path.isdir(MODEL_DIR):
    print(f"[WARNING] MODEL_DIR does not exist: {MODEL_DIR}")
else:
    try:
        files = os.listdir(MODEL_DIR)
        print(f"[INFO] MODEL_DIR={MODEL_DIR}, contains: {files}")
    except Exception as e:
        print(f"[WARNING] Could not list MODEL_DIR {MODEL_DIR}: {e}")

GENERAL_MODEL = None 
INFERENCE_MAX_DIM = 640 # For resizing high-res frames for faster inference

# Device auto-selection
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {DEVICE}")

# --- COCO CLASS MAPPING ---
# This map links user-friendly names to the corresponding COCO class IDs (0-indexed).

YOLO_COCO_CLASS_MAP = {
    # Class ID 0 is 'person' which we use for face/head redaction
    'People': [0],
    'Faces': [0], 
    
    # Vehicles (for your car detection issue)
    'Vehicles': [2, 3, 5, 7], # 2: car, 3: motorcycle, 5: bus, 7: truck
    'Cars': [2],
    'Trucks': [7],
    
    # Common PII objects / Screens
    'Cell phones': [68],
    'Laptops': [64],
    'Screens': [64, 68, 73], # Laptops, Cell Phones, TV Monitors
}

# --- 1. MODEL LOADING LOGIC ---

def _load_model_with_fallback(candidates, model_type):
    """Try loading multiple candidate models until one succeeds."""
    last_exc = None
    for name in candidates:
        path = os.path.join(MODEL_DIR, name) 
        if not os.path.isfile(path):
            print(f"[WARNING] Model file not found: {path}")
            continue
        try:
            print(f"[INFO] Attempting to load {model_type} model: {path}")
            model = YOLO(path)
            print(f"[INFO] Loaded {model_type} model: {path}")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            last_exc = e
            
    raise last_exc if last_exc else FileNotFoundError(f"No model found for {model_type}")

# Load models upon script initialization
try:
    GENERAL_MODEL = _load_model_with_fallback(['yolov9c.pt', 'yolov8n.pt'], 'general')
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load general model: {e}")

# --- 2. CORE UTILITY FUNCTIONS (Redaction Styles) ---

def clamp(v, a, b):
    """Utility function to clamp a value between min (a) and max (b)."""
    return max(a, min(b, v))

def apply_blur(frame, x1, y1, x2, y2):
    """Apply Gaussian Blur to the ROI with dynamic kernel size."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or x2 <= x1 or y2 <= y1:
        return frame
    
    width = x2 - x1
    ksize_dynamic = max(15, (width // 3)) 
    k = max(1, (ksize_dynamic // 2) * 2 + 1)
    
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame

def apply_pixelate(frame, x1, y1, x2, y2):
    """Apply Pixelation to the ROI with dynamic block size."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or x2 <= x1 or y2 <= y1:
        return frame
        
    h, w = roi.shape[:2]
    blocks_dynamic = max(6, (w // 10))
    blocks = max(1, min(blocks_dynamic, min(h, w)))
    
    temp = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixelated
    return frame

def apply_block(frame, x1, y1, x2, y2, color=(0, 0, 0)):
    """Apply solid black block to the ROI."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    return frame

def apply_redaction_style(frame, boxes, method='Black box', padding=0.15):
    """
    Redacts all bounding boxes in a frame using the chosen method and padding.
    """
    h, w = frame.shape[:2]
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # 1. Add padding
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        
        # Clamp coordinates to frame boundaries
        x1 = clamp(x1 - pad_x, 0, w)
        y1 = clamp(y1 - pad_y, 0, h)
        x2 = clamp(x2 + pad_x, 0, w)
        y2 = clamp(y2 + pad_y, 0, h)
        
        # 2. Apply chosen redaction method
        if method == 'Blur':
            frame = apply_blur(frame, x1, y1, x2, y2)
        elif method == 'Pixelate':
            frame = apply_pixelate(frame, x1, y1, x2, y2)
        elif method == 'Black box':
            frame = apply_block(frame, x1, y1, x2, y2)
            
    return frame

# --- 3. INFERENCE OPTIMIZATION UTILITIES ---

def _resize_for_inference(frame):
    """Resizes frame to a max dimension for faster YOLO inference."""
    orig_h, orig_w = frame.shape[:2]
    max_dim = max(orig_w, orig_h)
    
    if max_dim <= INFERENCE_MAX_DIM:
        return frame, 1.0, 1.0
        
    scale = INFERENCE_MAX_DIM / float(max_dim)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    
    if new_w <= 0 or new_h <= 0:
        return frame, 1.0, 1.0
        
    resized = cv2.resize(frame, (new_w, new_h))
    scale_x = new_w / float(orig_w)
    scale_y = new_h / float(orig_h)
    return resized, scale_x, scale_y

def _map_coords_to_original(xy_resized, scale_x, scale_y, orig_w, orig_h):
    """Maps coordinates from the resized inference frame back to the original frame size."""
    xy_list = None
    try:
        xy_list = xy_resized.cpu().numpy().tolist()
    except:
        xy_list = xy_resized

    x1 = int(clamp(round(xy_list[0] / (scale_x if scale_x else 1.0)), 0, orig_w))
    y1 = int(clamp(round(xy_list[1] / (scale_y if scale_y else 1.0)), 0, orig_h))
    x2 = int(clamp(round(xy_list[2] / (scale_x if scale_x else 1.0)), 0, orig_w))
    y2 = int(clamp(round(xy_list[3] / (scale_y if scale_y else 1.0)), 0, orig_h))
    
    return [x1, y1, x2, y2]


# --- 4. CATEGORY FILTERING ---

def is_category_selected(class_id: int, selected_categories: list) -> bool:
    """Checks if a detected class ID matches a user-selected category using the COCO map."""
    
    for category_name, class_ids in YOLO_COCO_CLASS_MAP.items():
        if category_name in selected_categories and class_id in class_ids:
            return True
            
    return False


# --- 5. MAIN VIDEO PROCESSING FUNCTION ---

# Renamed to process_video for clarity, assuming the Streamlit frontend 
# will be updated to import this name.
def process_video(input_path: str, output_path: str, settings: dict) -> bool:
    """Process video, redact selected objects/faces frame by frame."""
    if not GENERAL_MODEL:
        print("Error: YOLO models are not loaded.")
        return False

    # Retrieve settings from the dictionary
    selected_categories = settings.get('categories', [])
    redaction_style = settings.get('style', 'Black box')
    frame_skip = max(1, settings.get('frame_skip', 5)) 
    conf_threshold = settings.get('confidence', 0.1) 
    pad_fraction = settings.get('padding', 0.15) 
    # skip_category_filter is not used in the final app but is kept for backend testing
    skip_category_filter = settings.get('skip_category_filter', False)


    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return False

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    if not out.isOpened():
        print("Error: VideoWriter could not be opened. Check file path permissions.")
        cap.release()
        return False

    frame_counter = 0
    detection_log = {'person': 0, 'other_objects': 0}
    last_boxes = [] # Bounding box persistence storage

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_boxes = []

        # --- MODEL INFERENCE (Only run every Nth frame) ---
        if frame_counter % frame_skip == 0:
            
            frame_detections = {'person': 0, 'other_objects': 0}
            
            frame_input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame, sx, sy = _resize_for_inference(frame_input_rgb)

            # --- General Model Detection ---
            try:
                results = GENERAL_MODEL(resized_frame, verbose=False, device=DEVICE, conf=conf_threshold) 
                
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                        
                        if skip_category_filter or is_category_selected(class_id, selected_categories):
                            
                            class_name = GENERAL_MODEL.names.get(class_id, f"Class {class_id}")
                            
                            if class_id == 0: 
                                detection_log['person'] += 1
                                log_type = "PERSON/FACE"
                            else:
                                detection_log['other_objects'] += 1
                                log_type = "GENERAL"

                            original_coords = _map_coords_to_original(box.xyxy[0], sx, sy, orig_w, orig_h)
                            current_frame_boxes.append(original_coords)
                            
                            # Diagnostic Print
                            print(f"[{log_type}] F{frame_counter:04d}: {class_name} (ID {class_id}, Conf={conf:.2f}). Box: {original_coords}")

            except Exception as e:
                print(f"[ERROR] General model inference failed on frame {frame_counter}: {e}")
            
            last_boxes = current_frame_boxes
            
        else:
            current_frame_boxes = last_boxes
        
        # --- Apply Redaction ---
        if current_frame_boxes:
            frame = apply_redaction_style(
                frame, 
                current_frame_boxes, 
                method=redaction_style, 
                padding=pad_fraction
            )
        
        out.write(frame)
        frame_counter += 1

    cap.release()
    out.release()

    print("\n--- VIDEO PROCESSING SUMMARY ---")
    print(f"Total Frames Processed: {frame_counter}")
    print(f"Total Person/Face Detections (Class ID 0): {detection_log['person']}")
    print(f"Total General (Non-Person) Detections: {detection_log['other_objects']}")

    return True