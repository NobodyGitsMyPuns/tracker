import cv2
import sys
import os
import torch
from ultralytics import YOLO
import numpy as np
from esp32_drone_integration import ESP32DroneTracker
import threading
import queue
import time
from config import config

# Set environment variable to use the old PyTorch loading behavior
os.environ['TORCH_SERIALIZATION_WEIGHTS_ONLY'] = 'False'

# Global variables for threading (using config)
display_queue = queue.Queue(maxsize=config.DISPLAY_QUEUE_SIZE)
info_queue = queue.Queue(maxsize=config.INFO_QUEUE_SIZE)
key_queue = queue.Queue()
gui_running = True

# Digital zoom variables (using config)
zoom_factor = config.DEFAULT_ZOOM_FACTOR
zoom_center_x = 0.5 # Center of zoom (0-1 range)
zoom_center_y = 0.5 # Center of zoom (0-1 range)

def gui_thread():
 """Separate thread for GUI display and key handling"""
 global gui_running

 # Main camera window - RESIZABLE with zoom support (using config)
 cv2.namedWindow(config.MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
 cv2.resizeWindow(config.MAIN_WINDOW_NAME, config.MAIN_WINDOW_WIDTH, config.MAIN_WINDOW_HEIGHT)

 # Info panel window for data/logs (separate window)
 cv2.namedWindow(config.INFO_PANEL_NAME, cv2.WINDOW_NORMAL)
 cv2.resizeWindow(config.INFO_PANEL_NAME, config.INFO_PANEL_WIDTH, config.INFO_PANEL_HEIGHT)

 # Position windows side by side (using config)
 cv2.moveWindow(config.MAIN_WINDOW_NAME, config.MAIN_WINDOW_POS_X, config.MAIN_WINDOW_POS_Y)
 cv2.moveWindow(config.INFO_PANEL_NAME, config.INFO_PANEL_POS_X, config.INFO_PANEL_POS_Y)

 while gui_running:
 try:
 # Get latest frame (non-blocking)
 frame = None
 while not display_queue.empty():
 try:
 frame = display_queue.get_nowait()
 except queue.Empty:
 break

 if frame is not None:
 cv2.imshow(config.MAIN_WINDOW_NAME, frame)

 # Get latest info panel data (non-blocking)
 info_data = None
 while not info_queue.empty():
 try:
 info_data = info_queue.get_nowait()
 except queue.Empty:
 break

 if info_data is not None:
 # Create info panel image
 info_panel = create_info_panel(info_data)
 cv2.imshow(config.INFO_PANEL_NAME, info_panel)

 # Check for key presses
 key = cv2.waitKey(1) & 0xFF
 if key != 255 and key != 0:
 try:
 key_queue.put_nowait(key)
 except queue.Full:
 pass

 if key == ord('q') or key == 27:
 gui_running = False
 break

 except Exception as e:
 print(f"GUI Error: {e}")
 time.sleep(0.01)

 cv2.destroyAllWindows()

def create_info_panel(info_data):
 """Create info panel image with logs and system data"""
 panel_width = config.INFO_PANEL_WIDTH
 panel_height = config.INFO_PANEL_HEIGHT
 panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

 # Background color (dark gray)
 panel.fill(40)

 y_pos = 30
 line_height = 25

 # Title
 cv2.putText(panel, "DRONE TRACKER - SYSTEM INFO", (10, y_pos),
 cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_LARGE, config.COLOR_YELLOW, 2)
 y_pos += 40

 # Drone position data (if available)
 if 'drone_data' in info_data and info_data['drone_data']:
 data = info_data['drone_data']
 cv2.putText(panel, "DRONE TARGETING:", (10, y_pos),
 cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE_MEDIUM, config.COLOR_GREEN, 2)
 y_pos += line_height

 cv2.putText(panel, f"MOVE: {data['x_dir']} {data['angle_x']:.1f}°, {data['y_dir']} {data['angle_y']:.1f}°",
 (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 y_pos += line_height

 cv2.putText(panel, f"RAW: X={data['deviation_x']:+.0f}px, Y={data['deviation_y']:+.0f}px",
 (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 y_pos += 35

 # System logs
 if 'log_messages' in info_data:
 cv2.putText(panel, "SYSTEM LOG:", (10, y_pos),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
 y_pos += line_height

 for message in info_data['log_messages'][-10:]: # Last 10 messages
 if y_pos > panel_height - 20:
 break

 # Color code messages
 if "ERROR" in message or "FAILED" in message or "ERROR:" in message:
 color = (0, 0, 255) # Red
 elif "SUCCESS" in message or "SUCCESS:" in message:
 color = (0, 255, 0) # Green
 elif "WARNING" in message or "WARNING:" in message:
 color = (0, 255, 255) # Yellow
 else:
 color = (255, 255, 255) # White

 # Truncate long messages
 display_msg = message if len(message) < 70 else message[:67] + "..."
 cv2.putText(panel, display_msg, (10, y_pos),
 cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
 y_pos += 18

 # Controls help at bottom
 help_y = panel_height - 60
 cv2.putText(panel, "CONTROLS: WASD=5° | Arrows=15° | SPACE=Fire | C=Center | Q=Quit",
 (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

 cv2.putText(panel, "ZOOM: + (zoom in) | - (zoom out) | 0 (reset zoom)",
 (10, help_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

 return panel

def apply_digital_zoom(frame, zoom_factor, center_x, center_y):
 """Apply digital zoom to frame"""
 if zoom_factor <= 1.0:
 return frame

 height, width = frame.shape[:2]

 # Calculate crop dimensions
 crop_width = int(width / zoom_factor)
 crop_height = int(height / zoom_factor)

 # Calculate crop position (centered on zoom_center)
 start_x = int((width * center_x) - (crop_width / 2))
 start_y = int((height * center_y) - (crop_height / 2))

 # Ensure crop stays within frame bounds
 start_x = max(0, min(start_x, width - crop_width))
 start_y = max(0, min(start_y, height - crop_height))

 # Crop the frame
 cropped = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]

 # Resize back to original size
 zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

 return zoomed

def main():
 global gui_running

 # Load YOLOv8 model (this will download the model on first run)
 print("Loading YOLOv8 model...")

 # Monkey patch torch.load to use weights_only=False
 original_load = torch.load
 def patched_load(*args, **kwargs):
 kwargs.setdefault('weights_only', False)
 return original_load(*args, **kwargs)

 torch.load = patched_load

 try:
 # Check CUDA first
 if torch.cuda.is_available():
 print(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
 device = 'cuda'
 else:
 print("CUDA not available, using CPU")
 device = 'cpu'

 # Show model selection menu
 print("\n--- Select Detection Model ---")
 models = [
 ("yolov8n.pt", "YOLOv8n (nano)", "~3M", "Fastest, household objects"),
 ("yolov8s.pt", "YOLOv8s (small)", "~11M", "Good balance, household objects"),
 ("yolov8m.pt", "YOLOv8m (medium)", "~26M", "Better accuracy, household objects"),
 ("yolov8l.pt", "YOLOv8l (large)", "~43M", "High accuracy, household objects"),
 ("yolov8x.pt", "YOLOv8x (extra-large)", "~68M", "BEST accuracy, household objects"),
 ("yolov10n.pt", "YOLOv10n (nano)", "~3M", "Newer architecture, faster"),
 ("yolov10s.pt", "YOLOv10s (small)", "~8M", "Newer architecture, balanced"),
 ("yolov10m.pt", "YOLOv10m (medium)", "~16M", "Newer architecture, better"),
 ("yolov10l.pt", "YOLOv10l (large)", "~24M", "Newer architecture, high accuracy"),
 ("yolov10x.pt", "YOLOv10x (extra-large)", "~29M", "Newer architecture, BEST")
 ]

 for i, (file, name, params, desc) in enumerate(models):
 print(f"{i+1}. {name} - {params} parameters - {desc}")

 print(f"\n For tech environments, try YOLOv10 models (6-10) - newer architecture!")
 print("WARNING: Note: All YOLO models are trained on household objects, not electronics")

 while True:
 try:
 choice = input(f"\nSelect model (1-{len(models)}) [default: 10 for latest]: ").strip()
 if choice == "":
 choice = "10" # Default to YOLOv11x
 choice = int(choice)
 if 1 <= choice <= len(models):
 selected_model = models[choice-1]
 break
 else:
 print(f"Please enter a number between 1 and {len(models)}")
 except ValueError:
 print("Please enter a valid number")

 model_file, model_name, model_params, model_desc = selected_model
 print(f"\nLoading {model_name}...")

 # Try to load the selected model with fallback
 try:
 model = YOLO(model_file)
 model.to(device)
 except Exception as e:
 print(f"Error loading {model_file}: {e}")
 print("Falling back to YOLOv8x (known working model)...")
 model_file = "yolov8x.pt"
 model_name = "YOLOv8x (fallback)"
 model_params = "~68M"
 model_desc = "BEST accuracy, household objects"
 model = YOLO(model_file)
 model.to(device)
 finally:
 # Restore original torch.load
 torch.load = original_load

 # Print detailed model information
 print("Model loaded successfully!")
 print(f"Model file: {model_file}")
 print(f"Selected: {model_name} - {model_desc}")
 print(f"Model type: {type(model.model).__name__}")
 print(f"Model variant: {getattr(model.model, 'yaml', {}).get('nc', 'Unknown')} classes")
 print(f"Device: {model.device}")
 print(f"Model size: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M parameters")
 print(f"Architecture: {model_file.replace('.pt', '.yaml')}")
 print(f"Backbone layers: {len([m for m in model.model.modules() if 'Conv' in str(type(m))])} Conv layers")
 print(f"Input shape: {model.model.args.get('imgsz', 640) if hasattr(model.model, 'args') else 640}")
 print(f"Classes: {len(model.names)} total")

 # Show YOLOv8 model hierarchy and current selection
 print("\n--- YOLOv8 Model Hierarchy ---")
 hierarchy_models = [
 ("YOLOv8n (nano)", "~3M", "Fastest, least accurate"),
 ("YOLOv8s (small)", "~11M", "Good balance"),
 ("YOLOv8m (medium)", "~26M", "Better accuracy"),
 ("YOLOv8l (large)", "~43M", "High accuracy"),
 ("YOLOv8x (extra-large)", "~68M", "BEST accuracy")
 ]

 for i, (name, params, desc) in enumerate(hierarchy_models):
 if model_name.lower() in name.lower():
 print(f"{i+1}. {name} - {params} parameters - {desc} ← SELECTED")
 else:
 print(f"{i+1}. {name} - {params} parameters - {desc}")

 # Show the actual model architecture summary
 try:
 total_params = sum(p.numel() for p in model.model.parameters())
 trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
 print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")
 if "yolov8x" in model_name.lower():
 print("You are running the LARGEST and most accurate YOLOv8 model!")
 elif "yolov8n" in model_name.lower():
 print("You are running the FASTEST YOLOv8 model!")
 else:
 print(f"You are running {model_name}!")

 # Explain YOLO limitations and alternatives for tech environments
 print(f"\nWARNING: IMPORTANT: YOLO was trained on 80 common household objects:")
 print("SUCCESS: Good at: people, furniture, vehicles, animals, common electronics")
 print("ERROR: Bad at: Arduino boards, chips, cables, specific tech gear, lab equipment")
 print("Expect misclassifications: monitors→TV, small boxes→microwave, keyboards→piano")
 print(" Use this for general object detection, not precise tech identification")

 print(f"\n BETTER ALTERNATIVES for your tech environment:")
 print("1. CLIP (OpenAI) - Can detect any object you describe in text")
 print("2. DETIC - 21,000+ categories including electronics")
 print("3. GroundingDINO - Text-prompted object detection")
 print("4. Custom YOLO - Train on your own electronics dataset")
 print("5. SAM + CLIP - Segment everything + classify with text")
 print(" For now, YOLO gives you bounding boxes but wrong labels\n")
 except:
 pass

 # Initialize camera with DirectShow only (skip b1

 print("Trying DirectShow backend (skipping broken Media Foundation)...")
 for camera_index in range(5):
 try:
 cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
 if cap.isOpened():
 # Force specific settings before testing (LARGER RESOLUTION)
 # 4090 OPTIMIZED: High-res, high-FPS for smooth tracking
 cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # Full HD for 4090
 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Full HD for 4090
 cap.set(cv2.CAP_PROP_FPS, 60) # 60 FPS for ultra-smooth
 cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimal buffer for low latency
 cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

 # Give it a moment to initialize
 import time
 time.sleep(0.5)

 # Try multiple frame reads
 for attempt in range(5):
 ret, test_frame = cap.read()
 if ret and test_frame is not None:
 print(f"Camera working at index {camera_index} (attempt {attempt + 1})")
 break
 time.sleep(0.1)

 if ret and test_frame is not None:
 break
 else:
 print(f"Camera {camera_index} opened but can't read frames")
 cap.release()
 cap = None
 except Exception as e:
 print(f"Error with camera {camera_index}: {e}")
 if cap:
 cap.release()
 cap = None

 if cap is None or not cap.isOpened():
 print("Error: Could not open any camera with any backend")
 print("Make sure your camera isn't being used by another application")
 print("Try closing Zoom, Teams, or other camera apps and run again")
 sys.exit()

 print("Camera opened successfully!")
 print("DRONE-OPTIMIZED DETECTION MODE ACTIVE!")
 print(" Prioritizing: clock, bird, kite, frisbee, sports ball, donut, apple, orange")
 print("Lower confidence threshold (0.15) for aerial objects")
 print(" MOVEMENT DETECTION: Moving 'clock' = CONFIRMED DRONE!")
 print("Press 'q' to quit or 'ESC' to exit")
 print("Real-time drone detection starting...")

 frame_count = 0
 # Movement tracking for drone detection (using config)
 previous_positions = {} # Store previous positions by class
 movement_threshold = config.MOVEMENT_THRESHOLD

 # Enhanced tracking for moving drones (using config)
 drone_tracking_history = {} # Store last N positions for each drone type
 max_history = config.MAX_TRACKING_HISTORY

 # Camera field of view settings for angle calculations (from config)
 camera_fov_horizontal = config.CAMERA_FOV_HORIZONTAL
 camera_fov_vertical = config.CAMERA_FOV_VERTICAL

 # Get actual frame dimensions from camera (don't hardcode)
 ret, test_frame = cap.read()
 if ret and test_frame is not None:
 frame_height, frame_width = test_frame.shape[:2]
 print(f"Camera resolution: {frame_width}x{frame_height}")
 else:
 # Fallback dimensions if can't read frame
 frame_width = 1280
 frame_height = 720
 print(f"WARNING: Using fallback resolution: {frame_width}x{frame_height}")

 # On-screen logging system (using config)
 log_messages = [] # Store recent log messages
 max_log_messages = config.MAX_LOG_MESSAGES

 def add_log(message):
 """Add a log message to on-screen display"""
 import time
 timestamp = time.strftime("%H:%M:%S")
 log_messages.append(f"[{timestamp}] {message}")
 if len(log_messages) > max_log_messages:
 log_messages.pop(0) # Remove oldest message
 print(f"LOG: {message}") # Also print to console

 # Initialize ESP32 drone tracker (using config)
 print("Initializing ESP32 drone tracker...")
 esp32_tracker = ESP32DroneTracker() # Uses config.ESP32_IP

 # Add startup log messages
 add_log("System starting...")
 add_log(" ESP32 connection test...")
 if esp32_tracker.test_connection():
 add_log("SUCCESS: ESP32 connected successfully")
 else:
 add_log("ERROR: ESP32 connection FAILED - check network")

 # Manual control variables (from config)
 manual_step_size = config.MANUAL_STEP_SIZE
 manual_step_size_shift = config.MANUAL_STEP_SIZE_SHIFT

 print("Initializing threaded camera and GUI...")

 # Start GUI thread
 gui_thread_handle = threading.Thread(target=gui_thread)
 gui_thread_handle.daemon = True
 gui_thread_handle.start()

 # Initialize camera (using config)
 cap = cv2.VideoCapture(config.CAMERA_INDEX)
 cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

 # Main processing loop
 while gui_running:
 # Capture frame-by-frame with retry logic
 ret, frame = cap.read()

 if not ret:
 print("ERROR: Failed to grab frame")
 continue

 # Apply digital zoom if enabled
 global zoom_factor, zoom_center_x, zoom_center_y
 if zoom_factor > 1.0:
 frame = apply_digital_zoom(frame, zoom_factor, zoom_center_x, zoom_center_y)

 # Run YOLOv8 inference with configurable settings
 results = model(frame, verbose=False, conf=config.CONFIDENCE_THRESHOLD,
 iou=config.IOU_THRESHOLD, max_det=config.MAX_DETECTIONS)

 # Draw the results on the frame
 annotated_frame = results[0].plot()

 # Draw CENTER REFERENCE CROSSHAIR on camera (AFTER YOLO annotations)
 # Apply offset from config
 center_offset_x = int((config.CROSSHAIR_OFFSET_X / camera_fov_horizontal) * frame_width)
 center_offset_y = int((config.CROSSHAIR_OFFSET_Y / camera_fov_vertical) * frame_height)

 center_x = int(frame_width / 2) + center_offset_x
 center_y = int(frame_height / 2) + center_offset_y

 # Debug: Print crosshair position occasionally (using config)
 if frame_count % config.DEBUG_FRAME_INTERVAL == 0:
 print(f"Crosshair at: ({center_x}, {center_y}) | Frame: {frame_width}x{frame_height} | Offset: +{center_offset_x},+{center_offset_y}")

 # Colors for center crosshair (using config)
 green = config.COLOR_GREEN
 white = config.COLOR_WHITE

 # Draw center crosshair lines (make them thicker and more visible)
 cv2.line(annotated_frame, (center_x - 30, center_y), (center_x + 30, center_y), green, 3) # Horizontal line
 cv2.line(annotated_frame, (center_x, center_y - 30), (center_x, center_y + 30), green, 3) # Vertical line

 # Draw center dot (make it larger and more visible)
 cv2.circle(annotated_frame, (center_x, center_y), 8, white, -1) # White filled circle (larger)
 cv2.circle(annotated_frame, (center_x, center_y), 8, green, 3) # Green border (thicker)

 # Add center reference label
 cv2.putText(annotated_frame, "CENTER", (center_x - 30, center_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

 # Add bullseye targets for detected drones (using config)
 drone_classes = config.DRONE_CLASSES

 for i, box in enumerate(results[0].boxes):
 class_id = int(box.cls[0])
 confidence = float(box.conf[0])
 class_name = model.names[class_id]

 # Only draw bullseye for drone-like objects with ULTRA-LOW threshold
 if class_name in drone_classes and confidence > 0.05:
 # Get bounding box coordinates
 x1, y1, x2, y2 = box.xyxy[0].tolist()
 drone_center_x = int((x1 + x2) / 2)
 drone_center_y = int((y1 + y2) / 2)

 # Draw bullseye target centered on detected drone

 # Colors: BGR format (Blue, Green, Red)
 red = (0, 0, 255)
 white = (255, 255, 255)
 yellow = (0, 255, 255)

 # Draw concentric circles for bullseye effect
 cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 50, red, 3) # Outer red circle
 cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 35, white, 3) # Middle white circle
 cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 20, red, 3) # Inner red circle
 cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 10, yellow, 3) # Center yellow circle
 cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 3, red, -1) # Center red dot (filled)

 # Draw crosshairs
 cv2.line(annotated_frame, (drone_center_x - 60, drone_center_y), (drone_center_x - 50, drone_center_y), red, 3) # Left
 cv2.line(annotated_frame, (drone_center_x + 50, drone_center_y), (drone_center_x + 60, drone_center_y), red, 3) # Right
 cv2.line(annotated_frame, (drone_center_x, drone_center_y - 60), (drone_center_x, drone_center_y - 50), red, 3) # Top
 cv2.line(annotated_frame, (drone_center_x, drone_center_y + 50), (drone_center_x, drone_center_y + 60), red, 3) # Bottom

 # Calculate position data for on-screen display
 frame_center_x = frame_width / 2
 frame_center_y = frame_height / 2

 # Pixel offsets from center
 x_offset_pixels = drone_center_x - frame_center_x
 y_offset_pixels = drone_center_y - frame_center_y

 # Angle offsets from center
 angle_x = (x_offset_pixels / frame_width) * camera_fov_horizontal
 angle_y = -(y_offset_pixels / frame_height) * camera_fov_vertical # Negative because Y increases downward

 # Direction strings
 x_dir = "RIGHT" if x_offset_pixels > 0 else "LEFT"
 y_dir = "DOWN" if y_offset_pixels > 0 else "UP"

 # Add "ENEMY DRONE" label above the bullseye
 label_text = f"ENEMY DRONE ({class_name.upper()})"
 label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
 label_x = drone_center_x - label_size[0] // 2
 label_y = drone_center_y - 80

 # Draw label background
 cv2.rectangle(annotated_frame, (label_x - 5, label_y - 25), (label_x + label_size[0] + 5, label_y + 5), (0, 0, 0), -1)
 cv2.putText(annotated_frame, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, yellow, 2)

 # Store drone position data for top-of-screen display
 drone_position_data = {
 'x_dir': x_dir,
 'y_dir': y_dir,
 'x_offset': abs(x_offset_pixels),
 'y_offset': abs(y_offset_pixels),
 'angle_x': abs(angle_x),
 'angle_y': abs(angle_y),
 'deviation_x': x_offset_pixels,
 'deviation_y': y_offset_pixels
 }
 break # Only show data for the first detected drone

 # Prepare data for info panel (instead of overlaying on camera)
 info_data = {
 'drone_data': drone_position_data if 'drone_position_data' in locals() else None,
 'log_messages': log_messages
 }

 # Send info data to info panel (non-blocking)
 try:
 while not info_queue.empty():
 try:
 info_queue.get_nowait()
 except queue.Empty:
 break
 info_queue.put_nowait(info_data)
 except queue.Full:
 pass

 # Center crosshair now drawn after YOLO annotations (moved above)

 # Add control mode display at bottom of screen
 zoom_text = f" | ZOOM: ZOOM: {zoom_factor:.1f}x" if zoom_factor > 1.0 else ""
 status_text = f"CONTROLS: MANUAL CONTROL | WASD=5° | Arrow Keys=15° | SPACE=FIRE:FIRE | C=center{zoom_text} | H=help | Q=quit"

 # Draw control status at bottom
 text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
 status_x = 10
 status_y = frame_height - 20

 # Background for status
 cv2.rectangle(annotated_frame, (status_x - 5, status_y - 25), (status_x + text_size[0] + 10, status_y + 5), (0, 0, 0), -1)
 cv2.rectangle(annotated_frame, (status_x - 5, status_y - 25), (status_x + text_size[0] + 10, status_y + 5), (0, 255, 0), 2)

 # Status text
 cv2.putText(annotated_frame, status_text, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

 # Log messages now displayed in separate info panel window

 # Display detection information in terminal with full debugging
 if len(results[0].boxes) > 0:
 print(f"\nFIRE:FIRE:�� FRAME {frame_count} - FOUND {len(results[0].boxes)} OBJECTS! FIRE:FIRE:FIRE:")

 # Show ALL raw detections first
 all_detections = []
 for i, box in enumerate(results[0].boxes):
 class_id = int(box.cls[0])
 confidence = float(box.conf[0])
 class_name = model.names[class_id]

 # Get bounding box coordinates
 x1, y1, x2, y2 = box.xyxy[0].tolist()
 box_area = (x2 - x1) * (y2 - y1)

 # Calculate center point for movement tracking
 center_x = int((x1 + x2) / 2)
 center_y = int((y1 + y2) / 2)

 # Calculate angle offset from camera center
 frame_center_x = frame_width / 2
 frame_center_y = frame_height / 2

 # Calculate horizontal and vertical offsets in pixels
 offset_x_pixels = center_x - frame_center_x
 offset_y_pixels = center_y - frame_center_y

 # Convert pixel offsets to angles
 angle_x = (offset_x_pixels / frame_width) * camera_fov_horizontal
 angle_y = -(offset_y_pixels / frame_height) * camera_fov_vertical # Negative because Y increases downward

 # Direction descriptions
 horizontal_dir = "right" if angle_x > 0 else "left"
 vertical_dir = "up" if angle_y > 0 else "down"

 all_detections.append({
 'id': i,
 'class': class_name,
 'conf': confidence,
 'box': (int(x1), int(y1), int(x2), int(y2)),
 'area': int(box_area),
 'center': (center_x, center_y),
 'angle_x': angle_x,
 'angle_y': angle_y,
 'horizontal_dir': horizontal_dir,
 'vertical_dir': vertical_dir
 })

 # Sort by confidence (highest first)
 all_detections.sort(key=lambda x: x['conf'], reverse=True)

 print("Raw detections (sorted by confidence):")
 for det in all_detections:
 print(f" {det['class']}: {det['conf']:.3f} | box: {det['box']} | area: {det['area']}px")

 # Smart filtering with DRONE-BIASED detection for aerial objects
 final_detections = []
 drone_bias_corrections = {
 # Primary drone detection classes (YOLO often misclassifies drones as these)
 'clock': ' DRONE (round/circular aerial object)',
 'bird': ' DRONE (flying object)',
 'kite': ' DRONE (aerial flying object)',
 'frisbee': ' DRONE (round flying disc)',
 'sports ball': ' DRONE (round aerial object)',
 'donut': ' DRONE (round object with center hole)',
 'apple': ' DRONE (round compact object)',
 'orange': ' DRONE (round object)',
 'cell phone': ' DRONE (rectangular flying object)',
 'remote': ' DRONE (small rectangular controller-like object)',

 # Secondary objects (keep normal detection but note context)
 'microwave': 'small electronic device/equipment',
 'tv': 'monitor/display',
 'oven': 'large electronic equipment',
 'refrigerator': 'large metal object/equipment',
 'keyboard': 'piano/musical keyboard (if has keys)',
 'mouse': 'computer mouse or small device',
 'cup': 'cylindrical object (cup/candle/container)',
 'bottle': 'tall cylindrical object',
 'laptop': 'flat rectangular device',
 'book': 'flat rectangular object'
 }

 # Drone-prioritized detection with confidence boosting
 drone_classes = {'clock', 'bird', 'kite', 'frisbee', 'sports ball', 'donut', 'apple', 'orange', 'cell phone', 'remote'}

 for det in all_detections:
 class_name = det['class']
 conf = det['conf']
 area = det['area']
 center = det['center']
 angle_x = det['angle_x']
 angle_y = det['angle_y']
 horizontal_dir = det['horizontal_dir']
 vertical_dir = det['vertical_dir']

 # MOVEMENT DETECTION: Check if object moved significantly
 movement_detected = False
 movement_distance = 0
 if class_name in previous_positions:
 prev_center = previous_positions[class_name]
 movement_distance = ((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)**0.5
 movement_detected = movement_distance > movement_threshold

 # Update position tracking with history for better movement detection
 previous_positions[class_name] = center

 # Enhanced drone tracking - store position history
 if class_name in drone_classes:
 if class_name not in drone_tracking_history:
 drone_tracking_history[class_name] = []

 drone_tracking_history[class_name].append(center)

 # Keep only last N positions
 if len(drone_tracking_history[class_name]) > max_history:
 drone_tracking_history[class_name].pop(0)

 # Calculate movement consistency (for better drone detection)
 if len(drone_tracking_history[class_name]) >= 3:
 positions = drone_tracking_history[class_name]
 total_movement = 0
 for i in range(1, len(positions)):
 dx = positions[i][0] - positions[i-1][0]
 dy = positions[i][1] - positions[i-1][1]
 total_movement += (dx**2 + dy**2)**0.5

 # If consistently moving, boost confidence
 if total_movement > 30: # Moving consistently
 movement_detected = True
 movement_distance = total_movement

 # DRONE BIAS: ULTRA-LOW threshold for drone-like objects (catch everything!)
 min_confidence = 0.05 if class_name in drone_classes else 0.3

 if conf > min_confidence:
 # DRONE PRIORITY: Boost confidence display for potential drones
 if class_name in drone_classes:
 # Size analysis for drone detection
 size_analysis = ""
 if area < 1000:
 size_analysis = " (small/distant)"
 elif area < 5000:
 size_analysis = " (medium/close)"
 else:
 size_analysis = " (large/very close)"

 # POSITION/ANGLE INFORMATION for drone targeting
 position_info = f" [{horizontal_dir} {abs(angle_x):.1f}°, {vertical_dir} {abs(angle_y):.1f}°]"
 movement_info = f" [moved {movement_distance:.1f}px]" if movement_detected else ""

 if class_name == 'clock' and movement_detected:
 display_name = f"*** MOVING CLOCK = DRONE CONFIRMED! ***{size_analysis}{position_info}{movement_info}"
 elif class_name == 'clock':
 display_name = f"*** DRONE ALERT: CLOCK DETECTED! ***{size_analysis}{position_info} (watching for movement...)"
 elif class_name in {'bird', 'kite', 'frisbee'} and movement_detected:
 display_name = f"*** MOVING {class_name.upper()} = LIKELY DRONE! ***{size_analysis}{position_info}{movement_info}"
 elif class_name in {'bird', 'kite', 'frisbee'}:
 display_name = f"*** DRONE ALERT: {class_name.upper()} DETECTED! ***{size_analysis}{position_info}"
 elif movement_detected:
 display_name = f"*** MOVING {class_name.upper()} = POSSIBLE DRONE! ***{size_analysis}{position_info}{movement_info}"
 else:
 display_name = f"POSSIBLE DRONE as '{class_name}'{size_analysis}{position_info}"

 # Boost confidence display for drones (EXTRA boost for moving clocks!)
 if class_name == 'clock' and movement_detected:
 conf_boost = min(conf + 0.4, 1.0) # MAJOR boost for moving clock = confirmed drone
 elif movement_detected:
 conf_boost = min(conf + 0.3, 1.0) # Extra boost for any moving drone-like object
 else:
 conf_boost = min(conf + 0.2, 1.0) # Standard drone boost
 else:
 # Regular object detection
 size_corrections = ""
 if class_name == 'tv' and area < 20000:
 size_corrections = " (small - likely monitor)"
 elif class_name == 'microwave' and area < 5000:
 size_corrections = " (small - likely electronic device)"

 if class_name in drone_bias_corrections and conf < 0.8:
 display_name = f"{class_name} (probably {drone_bias_corrections[class_name]}){size_corrections}"
 else:
 display_name = f"{class_name}{size_corrections}"

 conf_boost = conf

 # Confidence labels with area info (use boosted confidence for display)
 if conf_boost > 0.8:
 conf_label = f" [{area}px²]"
 elif conf_boost > 0.6:
 conf_label = f" ? [{area}px²]"
 else:
 conf_label = f" deg [{area}px²]"

 final_detections.append(f"{display_name}: {conf:.2f}{conf_label}")

 # SUPER OBVIOUS DRONE STATUS ALERTS!
 drone_detected = any(d['class'] in drone_classes for d in all_detections if d['conf'] > 0.05)
 if drone_detected:
 print("DRONE DETECTION ACTIVE! WATCHING FOR AERIAL OBJECTS! ")
 print("PRIORITY TARGETS: clock, bird, kite, frisbee, sports ball, etc.")
 elif any(d['class'] in {'microwave', 'tv', 'oven', 'refrigerator'} for d in all_detections):
 print("NOTE: YOLO was trained on household items, not tech equipment!")

 if final_detections:
 print("=" * 80)
 print("DETECTED OBJECTS:")
 for detection in final_detections:
 print(f" {detection}")
 print("=" * 80)
 else:
 print("ZOOM: SCANNING... No confident detections (try moving drone closer or into better lighting)")
 else:
 if frame_count % 30 == 0: # Print every 30 frames
 print(f"Frame {frame_count}: No objects detected at all")

 # Send processed frame to GUI thread (non-blocking)
 try:
 # Clear old frames
 while not display_queue.empty():
 try:
 display_queue.get_nowait()
 except queue.Empty:
 break
 display_queue.put_nowait(annotated_frame)
 except queue.Full:
 pass

 # Check for key presses from GUI thread (non-blocking)
 key = None
 try:
 key = key_queue.get_nowait()
 except queue.Empty:
 pass

 if key is not None:
 # Process movement commands (same as before but non-blocking)
 if key != 255 and key != 0:
 add_log(f"�� Key pressed: {key} (char: '{chr(key) if 32 <= key <= 126 else '?'}')")

 # Manual movement controls (always active)
 if esp32_tracker:
 moved = False

 # Choose step size based on key type
 # WASD = 5° (fine), Arrow Keys = 15° (coarse)
 if key in [ord('w'), ord('s'), ord('a'), ord('d')]:
 current_step = manual_step_size # 5° for WASD
 key_type = "WASD"
 else:
 current_step = manual_step_size_shift # 15° for Arrow Keys
 key_type = "ARROW"

 try:
 if key == 82 or key == ord('w'): # Up arrow or W - Tilt up
 step_text = f"{current_step}° ({key_type})"
 add_log(f"UP: Tilt UP {step_text}")

 # Use ESP32 direct endpoints for reliable movement (FIXED)
 try:
 import requests
 response = requests.get(f"http://10.0.0.70/up?step={current_step}", timeout=1) # FIXED: W should go UP
 if response.status_code == 200:
 add_log(f"SUCCESS: {response.text}")
 else:
 add_log(f"ERROR: Tilt UP - HTTP {response.status_code}")
 except Exception as e:
 add_log(f"ERROR: Tilt UP - {str(e)[:30]}")
 moved = True

 elif key == 84 or key == ord('s'): # Down arrow or S - Tilt down
 step_text = f"{current_step}° ({key_type})"
 add_log(f"DOWN: Tilt DOWN {step_text}")

 try:
 response = requests.get(f"http://10.0.0.70/down?step={current_step}", timeout=1) # FIXED: S should go DOWN
 if response.status_code == 200:
 add_log(f"SUCCESS: {response.text}")
 else:
 add_log(f"ERROR: Tilt DOWN - HTTP {response.status_code}")
 except Exception as e:
 add_log(f"ERROR: Tilt DOWN - {str(e)[:30]}")
 moved = True

 elif key == 81 or key == ord('a'): # Left arrow or A - Pan left
 step_text = f"{current_step}° ({key_type})"
 add_log(f"LEFT: Pan LEFT {step_text}")

 try:
 response = requests.get(f"http://10.0.0.70/right?step={current_step}", timeout=1) # INVERTED
 if response.status_code == 200:
 add_log(f"SUCCESS: {response.text}")
 else:
 add_log(f"ERROR: Pan LEFT - HTTP {response.status_code}")
 except Exception as e:
 add_log(f"ERROR: Pan LEFT - {str(e)[:30]}")
 moved = True

 elif key == 83 or key == ord('d'): # Right arrow or D - Pan right
 step_text = f"{current_step}° ({key_type})"
 add_log(f"RIGHT: Pan RIGHT {step_text}")

 try:
 response = requests.get(f"http://10.0.0.70/left?step={current_step}", timeout=1) # INVERTED
 if response.status_code == 200:
 add_log(f"SUCCESS: {response.text}")
 else:
 add_log(f"ERROR: Pan RIGHT - HTTP {response.status_code}")
 except Exception as e:
 add_log(f"ERROR: Pan RIGHT - {str(e)[:30]}")
 moved = True
 except Exception as e:
 add_log(f"ERROR: Movement ERROR: {str(e)[:50]}")

 # System controls
 if key == ord('c'): # 'C' key - Center servos
 if esp32_tracker:
 try:
 add_log("Centering servos...")
 success = esp32_tracker.center_servos()
 if success:
 add_log("SUCCESS: Servos centered - SUCCESS")
 else:
 add_log("ERROR: Center servos - FAILED")
 except Exception as e:
 add_log(f"ERROR: Center ERROR: {str(e)[:50]}")

 if key == ord('x'): # 'X' key - Stop all movement
 if esp32_tracker:
 try:
 add_log("STOP: Stopping movement...")
 success = esp32_tracker.stop_movement()
 if success:
 add_log("SUCCESS: Movement stopped - SUCCESS")
 else:
 add_log("ERROR: Stop movement - FAILED")
 except Exception as e:
 add_log(f"ERROR: Stop ERROR: {str(e)[:50]}")

 if key == 32 or key == ord(' '): # Spacebar - Fire blaster! (try both codes)
 add_log(f"ZOOM: SPACEBAR detected! Key code: {key}")
 if esp32_tracker:
 add_log(" FIRING BLASTER...")
 try:
 import requests
 response = requests.get("http://10.0.0.70/fire", timeout=2)
 if response.status_code == 200:
 add_log("FIRE: BLASTER FIRED! (150ms)")
 else:
 add_log(f"ERROR: Fire FAILED - HTTP {response.status_code}")
 except Exception as e:
 add_log(f"ERROR: Fire FAILED - {str(e)[:30]}")
 else:
 add_log("ERROR: No ESP32 tracker available")

 if key == ord('+') or key == ord('='): # '+' key - Zoom in
 zoom_factor = min(zoom_factor * config.ZOOM_STEP, config.MAX_ZOOM_FACTOR)
 add_log(f"ZOOM: Zoom IN: {zoom_factor:.1f}x")

 elif key == ord('-') or key == ord('_'): # '-' key - Zoom out
 zoom_factor = max(zoom_factor / config.ZOOM_STEP, config.DEFAULT_ZOOM_FACTOR)
 if zoom_factor == config.DEFAULT_ZOOM_FACTOR:
 add_log("ZOOM: Zoom RESET: 1.0x (no zoom)")
 else:
 add_log(f"ZOOM: Zoom OUT: {zoom_factor:.1f}x")

 elif key == ord('0'): # '0' key - Reset zoom
 zoom_factor = config.DEFAULT_ZOOM_FACTOR
 add_log("ZOOM: Zoom RESET: 1.0x")

 elif key == ord('h'): # 'H' key - Show help
 print("\n" + "="*50)
 print("CONTROLS: MANUAL CAMERA CONTROL:")
 print("="*50)
 print(" MOVEMENT:")
 print(" WASD - Fine movement (5°)")
 print(" Arrow Keys - Coarse movement (15°)")
 print(" ↑/W - Tilt UP ↓/S - Tilt DOWN")
 print(" ←/A - Pan LEFT →/D - Pan RIGHT")
 print("")
 print("ZOOM: DIGITAL ZOOM:")
 print(" + - Zoom IN (up to 5x)")
 print(" - - Zoom OUT")
 print(" 0 - Reset zoom to 1x")
 print("")
 print("SYSTEM: SYSTEM:")
 print(" C - Center servos")
 print(" X - Stop movement")
 print(" SPACEBAR - FIRE: FIRE BLASTER! (150ms)")
 print(" H - Show this help")
 print(" Q - Quit")
 print("="*50)
 print("Drones will be detected and highlighted")
 print("CONTROLS: Use manual controls to aim at targets")
 print("="*50)

 # Small sleep to prevent CPU overload
 time.sleep(0.001) # 1ms - allows ~1000 FPS processing

 frame_count += 1

 # Cleanup
 cap.release()
 cv2.destroyAllWindows()

if __name__ == "__main__":
 main()