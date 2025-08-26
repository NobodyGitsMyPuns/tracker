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
from ai_parallax_correction import get_ai_crosshair_position, AIParallaxCorrector

# Set environment variable to use the old PyTorch loading behavior
os.environ['TORCH_SERIALIZATION_WEIGHTS_ONLY'] = 'False'

# Global variables for threading (using config)
display_queue = queue.Queue(maxsize=config.DISPLAY_QUEUE_SIZE)
info_queue = queue.Queue(maxsize=config.INFO_QUEUE_SIZE)
key_queue = queue.Queue()
gui_running = True

# NEW: AI tracking thread queues
ai_tracking_queue = queue.Queue(maxsize=5)  # YOLO results queue for AI thread
servo_command_queue = queue.Queue(maxsize=10)  # Servo commands queue

# Digital zoom variables (using config)
zoom_factor = config.DEFAULT_ZOOM_FACTOR
zoom_center_x = 0.5  # Center of zoom (0-1 range)
zoom_center_y = 0.5  # Center of zoom (0-1 range)

# Target selection variables (for TAB key functionality)
selected_target_x = None
selected_target_y = None
target_locked = False
red_crosshair_active = False
red_crosshair_x = None
red_crosshair_y = None

# Manual parallax calibration offsets (adjust these to fine-tune fire control)
# User has to manually aim LOW and RIGHT to hit target
# So red crosshair needs to be UP and LEFT from where AI puts it
parallax_offset_x = -150  # NEGATIVE = move red crosshair LEFT
parallax_offset_y = -250  # NEGATIVE = move red crosshair UP

# AI tracking disabled for stability
ai_tracking_active = False  # Always keep this off
ai_tracking_thread_running = True

def ai_tracking_thread():
    """Dedicated AI tracking thread - runs independently for responsive servo control"""
    global ai_tracking_thread_running, ai_tracking_active
    
    print("🤖 AI Tracking thread started")
    
    while ai_tracking_thread_running:
        try:
            # Get latest YOLO results (non-blocking with timeout)
            try:
                tracking_data = ai_tracking_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if tracking_data is None:  # Shutdown signal
                break
                
            results, frame_width, frame_height, zoom_factor, frame_count = tracking_data
            
            # Only track if AI tracking is active
            if not ai_tracking_active:
                continue
                
            # Calculate AI crosshair position
            ai_crosshair_x, ai_crosshair_y, ai_debug = get_ai_crosshair_position(
                results, frame_width, frame_height, zoom_factor
            )
            
            # Check if we have a valid target
            if ai_debug.get('status') == 'ai_corrected':
                # Calculate deviation from center for servo movement
                center_x = frame_width // 2
                center_y = frame_height // 2
                
                x_deviation = ai_crosshair_x - center_x
                y_deviation = ai_crosshair_y - center_y
                
                # Only move if deviation is significant (reduce jitter)
                move_threshold = 15  # pixels - reduced for more responsive tracking
                
                if abs(x_deviation) > move_threshold or abs(y_deviation) > move_threshold:
                    # Convert to angles
                    camera_fov_horizontal = config.CAMERA_FOV_HORIZONTAL / zoom_factor
                    camera_fov_vertical = config.CAMERA_FOV_VERTICAL / zoom_factor
                    
                    angle_x = (x_deviation / frame_width) * camera_fov_horizontal
                    angle_y = -(y_deviation / frame_height) * camera_fov_vertical
                    
                    # Create servo command
                    servo_cmd = {
                        'type': 'track',
                        'angle_x': angle_x,
                        'angle_y': angle_y,
                        'target_class': ai_debug.get('target_class', 'unknown'),
                        'confidence': ai_debug.get('confidence', 0),
                        'distance': ai_debug.get('estimated_distance_m', 0)
                    }
                    
                    # Send to servo thread (non-blocking)
                    try:
                        servo_command_queue.put_nowait(servo_cmd)
                    except queue.Full:
                        # Drop command if queue is full (prevent backup)
                        pass
                        
        except Exception as e:
            print(f"AI Tracking Error: {e}")
            time.sleep(0.01)
    
    print("🤖 AI Tracking thread stopped")

def servo_command_thread():
    """Dedicated servo command thread - handles all ESP32 communication"""
    import requests
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    import threading
    
    print("🎯 Servo Command thread started")
    print(f"🔗 Connecting to ESP32 at: {config.ESP32_IP}")
    
    last_command_time = 0
    command_rate_limit = 0.03  # 33 commands/sec - faster response
    
    # Create a session for connection reuse (faster)
    session = requests.Session()
    
    # Thread pool for async HTTP requests
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ServoHTTP")
    
    # Test connection first
    def test_connection():
        try:
            response = session.get(f"http://{config.ESP32_IP}/status", timeout=2.0)
            if response.status_code == 200:
                print(f"✅ ESP32 connection OK: {response.text.strip()}")
                return True
        except Exception as e:
            print(f"⚠️ ESP32 connection test failed: {str(e)[:50]}")
            print(f"🔧 Make sure ESP32 is powered on and connected to WiFi")
            print(f"🔧 Check ESP32_IP in config: {config.ESP32_IP}")
        return False
    
    # Test connection on startup
    connection_ok = test_connection()
    
    def send_http_request(url, timeout=1.0, log_success=True, log_error=True):
        """Send HTTP request asynchronously"""
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200 and log_success:
                print(f"✅ {response.text.strip()}")
                return True
            elif response.status_code != 200 and log_error:
                print(f"❌ HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectTimeout:
            if log_error:
                print(f"❌ Connection timeout to {config.ESP32_IP}")
        except requests.exceptions.ConnectionError:
            if log_error:
                print(f"❌ Cannot connect to ESP32 at {config.ESP32_IP}")
        except Exception as e:
            if log_error:
                print(f"❌ {str(e)[:40]}...")
        return False
    
    while ai_tracking_thread_running:
        try:
            # Get servo command (blocking with timeout)
            try:
                cmd = servo_command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if cmd is None:  # Shutdown signal
                break
                
            # Skip commands if no connection (for AI tracking)
            if not connection_ok and cmd['type'] == 'track':
                continue
                
            # Rate limiting for servo protection
            current_time = time.time()
            if current_time - last_command_time < command_rate_limit:
                time.sleep(command_rate_limit - (current_time - last_command_time))
            
            # Execute command based on type
            if cmd['type'] == 'track':
                angle_x = cmd['angle_x']
                angle_y = cmd['angle_y']
                
                # Send movement commands (larger steps for faster tracking)
                max_step = 5.0  # degrees - faster tracking response
                
                # Fire and forget - don't wait for responses during tracking
                if abs(angle_x) > 1.0:  # Reasonable threshold
                    step_x = min(abs(angle_x), max_step)
                    direction = "left" if angle_x > 0 else "right"
                    url = f"http://{config.ESP32_IP}/{direction}?step={step_x:.1f}"
                    executor.submit(send_http_request, url, 0.2, False, False)  # Fast timeout, no logging
                
                if abs(angle_y) > 1.0:  # Reasonable threshold
                    step_y = min(abs(angle_y), max_step)
                    direction = "up" if angle_y > 0 else "down"
                    url = f"http://{config.ESP32_IP}/{direction}?step={step_y:.1f}"
                    executor.submit(send_http_request, url, 0.2, False, False)  # Fast timeout, no logging
                        
            elif cmd['type'] == 'manual':
                # Manual movement commands
                direction = cmd['direction']
                step = cmd['step']
                url = f"http://{config.ESP32_IP}/{direction}?step={step}"
                
                print(f"🔧 DEBUG: Sending to ESP32: {url} (step={step})")
                
                # Submit and wait briefly for result (manual commands should provide feedback)
                future = executor.submit(send_http_request, url, 1.0, True, True)
                try:
                    result = future.result(timeout=1.5)  # Wait up to 1.5s for manual commands
                    if result:
                        connection_ok = True  # Mark connection as working
                except FutureTimeoutError:
                    print(f"❌ Manual {direction} - timeout")
                    connection_ok = False
                    
            elif cmd['type'] == 'center':
                # Center command
                url = f"http://{config.ESP32_IP}/center"
                future = executor.submit(send_http_request, url, 2.0, True, True)
                try:
                    result = future.result(timeout=2.5)
                    if result:
                        connection_ok = True
                except FutureTimeoutError:
                    print("❌ Center - timeout")
                    connection_ok = False
                    
            elif cmd['type'] == 'fire':
                # Fire command with longer timeout
                def fire_request():
                    try:
                        response = session.get(f"http://{config.ESP32_IP}/fire", timeout=3.0)
                        if response.status_code == 200:
                            print("🔥 BLASTER FIRED!")
                            return True
                        else:
                            print(f"❌ Fire failed - HTTP {response.status_code}")
                            return False
                    except Exception as e:
                        print(f"❌ Fire failed - {str(e)[:40]}")
                        return False
                
                future = executor.submit(fire_request)
                try:
                    result = future.result(timeout=4.0)
                    if result:
                        connection_ok = True
                except FutureTimeoutError:
                    print("❌ Fire - timeout")
                    connection_ok = False
            
            last_command_time = time.time()
            
        except Exception as e:
            print(f"Servo Command Error: {e}")
            time.sleep(0.01)
    
    # Cleanup
    executor.shutdown(wait=False)  # Don't wait for pending requests
    session.close()
    print("🎯 Servo Command thread stopped")

def gui_thread():
    """Separate thread for GUI display and key handling"""
    global gui_running
    
    # ... existing gui_thread code ...
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

def apply_digital_zoom(frame, zoom_factor, center_x=0.5, center_y=0.5):
    """Apply digital zoom to frame by cropping and resizing"""
    if zoom_factor <= 1.0:
        return frame
    
    height, width = frame.shape[:2]
    
    # Calculate crop dimensions
    crop_width = int(width / zoom_factor)
    crop_height = int(height / zoom_factor)
    
    # Calculate crop position (center_x and center_y are 0-1 range)
    crop_x = int((width - crop_width) * center_x)
    crop_y = int((height - crop_height) * center_y)
    
    # Ensure crop stays within bounds
    crop_x = max(0, min(crop_x, width - crop_width))
    crop_y = max(0, min(crop_y, height - crop_height))
    
    # Crop the frame
    cropped = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    
    # Resize back to original dimensions
    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return zoomed

def create_info_panel(info_data):
    """Create info panel showing system status, logs, and debug info"""
    # Panel dimensions from config
    panel_width = config.INFO_PANEL_WIDTH
    panel_height = config.INFO_PANEL_HEIGHT
    
    # Create black background
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # Colors
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    blue = (255, 0, 0)
    
    y_pos = 30
    line_height = 25
    
    # Title
    cv2.putText(panel, "🚁 DRONE TRACKER INFO", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, green, 2)
    y_pos += line_height * 2
    
    # AI Status
    ai_status = "🤖 AI TRACKING: ON" if ai_tracking_active else "🤖 AI TRACKING: OFF"
    cv2.putText(panel, ai_status, (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, green if ai_tracking_active else red, 2)
    y_pos += line_height
    
    # Zoom status
    global zoom_factor
    zoom_text = f"🔍 ZOOM: {zoom_factor:.1f}x"
    cv2.putText(panel, zoom_text, (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
    y_pos += line_height * 2
    
    # Drone data section
    drone_data = info_data.get('drone_data')
    if drone_data:
        cv2.putText(panel, "🎯 TARGET DATA:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
        y_pos += line_height
        
        for key, value in drone_data.items():
            text = f"  {key}: {value}"
            cv2.putText(panel, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
            y_pos += line_height
    else:
        cv2.putText(panel, "🎯 NO TARGET DETECTED", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        y_pos += line_height
    
    y_pos += line_height
    
    # AI Debug section
    ai_debug = info_data.get('ai_debug', {})
    if ai_debug:
        cv2.putText(panel, "🤖 AI DEBUG:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2)
        y_pos += line_height
        
        status = ai_debug.get('status', 'unknown')
        cv2.putText(panel, f"  Status: {status}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
        y_pos += line_height
        
        if 'target_class' in ai_debug:
            cv2.putText(panel, f"  Class: {ai_debug['target_class']}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
            y_pos += line_height
            
        if 'confidence' in ai_debug:
            cv2.putText(panel, f"  Conf: {ai_debug['confidence']:.2f}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
            y_pos += line_height
    
    y_pos += line_height
    
    # Log messages section
    log_messages = info_data.get('log_messages', [])
    if log_messages:
        cv2.putText(panel, "📋 RECENT LOGS:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)
        y_pos += line_height
        
        # Show last 10 messages (or whatever fits)
        max_logs = min(len(log_messages), (panel_height - y_pos - 30) // 20)
        start_idx = max(0, len(log_messages) - max_logs)
        
        for i in range(start_idx, len(log_messages)):
            msg = log_messages[i]
            # Truncate long messages
            if len(msg) > 45:
                msg = msg[:42] + "..."
            cv2.putText(panel, msg, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1)
            y_pos += 20
    
    # Controls reminder at bottom
    if y_pos < panel_height - 60:
        y_pos = panel_height - 60
        cv2.putText(panel, "🎮 CONTROLS:", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
        y_pos += 15
        cv2.putText(panel, "T=Toggle AI, WASD=2°, Arrows=15°", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1)
        y_pos += 15
        cv2.putText(panel, "SPACE=Fire, C=Center, H=Help", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, white, 1)
    
    return panel

def main():
    global gui_running, ai_tracking_active, ai_tracking_thread_running
    global red_crosshair_active, red_crosshair_x, red_crosshair_y
    
    # ... existing model loading code ...
    
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
        
        # Show model selection menu (all available models)
        print("\n--- Select Detection Model ---")
        models = [
            ("yolov8n.pt", "YOLOv8n (nano)", "~6M", "Fastest, household objects"),
            ("yolov8s.pt", "YOLOv8s (small)", "~22M", "Good balance, household objects"),
            ("yolov8m.pt", "YOLOv8m (medium)", "~52M", "Better accuracy, household objects"),
            ("yolov8l.pt", "YOLOv8l (large)", "~87M", "High accuracy, household objects"),
            ("yolov8x.pt", "YOLOv8x (extra-large)", "~136M", "BEST accuracy, household objects"),
            ("yolov10n.pt", "YOLOv10n (nano)", "~11M", "Newer architecture, faster"),
            ("yolov10s.pt", "YOLOv10s (small)", "~32M", "Newer architecture, balanced"),
            ("yolov10m.pt", "YOLOv10m (medium)", "~66M", "Newer architecture, better"),
            ("yolov10l.pt", "YOLOv10l (large)", "~104M", "Newer architecture, high accuracy"),
            ("yolov10x.pt", "YOLOv10x (extra-large)", "~26M", "Newer architecture, efficient")
        ]
        
        for i, (file, name, params, desc) in enumerate(models):
            print(f"{i+1}. {name} - {params} parameters - {desc}")
        
        print(f"\n💡 Recommended: YOLOv8x (#4) for best drone detection!")
        print("⚠️  Note: All YOLO models are trained on household objects, not electronics")
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(models)}) [default: 10 for YOLOv8x]: ").strip()
                if choice == "":
                    choice = "10"  # Default to YOLOv8x
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
    
    print("Model loaded successfully!")
    print(f"🚁 THREADED DRONE TRACKING MODE ACTIVE!")
    print("🤖 AI tracking runs in separate thread for maximum responsiveness")
    print("🎯 Servo commands run in dedicated thread with 20Hz update rate")
    
    # ... existing camera initialization code ...
    
    # Force specific camera index (change this number to your desired camera)
    FORCED_CAMERA_INDEX = 0  # Change this to 1, 2, 3, etc. for other cameras
    
    print(f"Using forced camera index: {FORCED_CAMERA_INDEX}")
    cap = None
    try:
        cap = cv2.VideoCapture(FORCED_CAMERA_INDEX, cv2.CAP_DSHOW)
        if cap.isOpened():
                # Force specific settings before testing (LARGER RESOLUTION)
                # 4090 OPTIMIZED: High-res, high-FPS for smooth tracking
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Full HD for 4090
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD for 4090
                cap.set(cv2.CAP_PROP_FPS, 60)             # 60 FPS for ultra-smooth
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Minimal buffer for low latency
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
    
    frame_count = 0
    
    # Get actual frame dimensions from camera (don't hardcode)
    ret, test_frame = cap.read()
    if ret and test_frame is not None:
        frame_height, frame_width = test_frame.shape[:2]
        print(f"📹 Camera resolution: {frame_width}x{frame_height}")
    else:
        # Fallback dimensions if can't read frame
        frame_width = 1280
        frame_height = 720
        print(f"⚠️ Using fallback resolution: {frame_width}x{frame_height}")
    
    # On-screen logging system (using config)
    log_messages = []              # Store recent log messages
    max_log_messages = config.MAX_LOG_MESSAGES
    
    def add_log(message):
        """Add a log message to on-screen display"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        log_messages.append(f"[{timestamp}] {message}")
        if len(log_messages) > max_log_messages:
            log_messages.pop(0)  # Remove oldest message
        print(f"LOG: {message}")  # Also print to console
    
    print("🚁 Initializing THREADED drone tracker...")
    
    # Add startup log messages
    add_log("🚁 THREADED system starting...")
    add_log("🤖 AI tracking thread initializing...")
    add_log("🎯 Servo command thread initializing...")
    
    print("Initializing AI parallax correction...")
    # Initialize AI parallax corrector with config
    import ai_parallax_correction
    ai_parallax_correction.ai_parallax = AIParallaxCorrector()
    ai_parallax_correction.ai_parallax.drone_only_mode = config.DRONE_ONLY_MODE
    ai_parallax_correction.ai_parallax.min_confidence_drone = config.MIN_DRONE_CONFIDENCE
    ai_parallax_correction.ai_parallax.smoothing_factor = config.CROSSHAIR_SMOOTHING
    
    # Start all threads
    print("🚁 Starting threaded camera, AI tracking, servo control, and GUI...")
    
    # Start AI tracking thread
    ai_thread = threading.Thread(target=ai_tracking_thread, daemon=True)
    ai_thread.start()
    add_log("🤖 AI tracking thread started")
    
    # Start servo command thread  
    servo_thread = threading.Thread(target=servo_command_thread, daemon=True)
    servo_thread.start()
    add_log("🎯 Servo command thread started")
    
    # Start GUI thread
    gui_thread_handle = threading.Thread(target=gui_thread, daemon=True)
    gui_thread_handle.start()
    add_log("🖥️ GUI thread started")
    
    print("✅ ALL THREADS RUNNING - Maximum performance mode active!")
    
    # Main processing loop - NOW ONLY HANDLES CAMERA AND YOLO
    while gui_running:
        # Capture frame-by-frame with retry logic
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            continue
        
        # Apply digital zoom if enabled
        global zoom_factor, zoom_center_x, zoom_center_y
        if zoom_factor > 1.0:
            frame = apply_digital_zoom(frame, zoom_factor, zoom_center_x, zoom_center_y)
        
        # Run YOLOv8 inference with configurable settings
        results = model(frame, verbose=False, conf=config.CONFIDENCE_THRESHOLD, 
                       iou=config.IOU_THRESHOLD, max_det=config.MAX_DETECTIONS)
        
        # Send YOLO results to AI tracking thread (non-blocking)
        tracking_data = (results[0], frame_width, frame_height, zoom_factor, frame_count)
        try:
            # Clear old data and send fresh data
            while not ai_tracking_queue.empty():
                try:
                    ai_tracking_queue.get_nowait()
                except queue.Empty:
                    break
            ai_tracking_queue.put_nowait(tracking_data)
        except queue.Full:
            pass  # Drop frame if queue is full
        
        # ... existing frame processing and display code (without AI tracking) ...
        
        # DON'T draw all results - we'll draw only drones manually
        annotated_frame = frame.copy()  # Start with clean frame
        
        # Calculate crosshair positions (but don't draw them yet - draw them LAST!)
        static_center_x = int(frame_width / 2)
        static_center_y = int(frame_height / 2)
        
        # Colors for dual crosshair system
        green = config.COLOR_GREEN
        white = config.COLOR_WHITE
        red = config.COLOR_RED
        yellow = config.COLOR_YELLOW
        blue = config.COLOR_BLUE
        
        # MANUALLY DRAW ONLY DRONE OBJECTS (expanded list for better detection!)
        strict_drone_classes = {
            # Primary drone-like objects
            'clock', 'bird', 'kite', 'frisbee', 'sports ball', 'remote',
            # Additional objects that drones might be misclassified as
            'donut', 'apple', 'orange', 'cell phone', 'mouse', 'cup', 'bottle'
        }
        
        # Initialize drone position variables (default to no drone detected)
        drone_center_x = None
        drone_center_y = None
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                # STRICT FILTER: Only draw actual drone objects with LOWER confidence (catch more drones!)
                if class_name in strict_drone_classes and confidence > 0.15:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    drone_center_x = int((x1 + x2) / 2)
                    drone_center_y = int((y1 + y2) / 2)
                    
                    # Draw bounding box for drone
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                    
                    # Draw drone label
                    label = f"{class_name.upper()}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
                    
                    # Draw bullseye target centered on detected drone
                    bullseye_color = (0, 255, 255)  # Yellow bullseye
                    cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 30, bullseye_color, 2)
                    cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 20, bullseye_color, 2)
                    cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 10, bullseye_color, 2)
                    cv2.circle(annotated_frame, (drone_center_x, drone_center_y), 3, bullseye_color, -1)
        
        # ... rest of existing display and key handling code ...
        
        # Display status
        zoom_text = f" | 🔍 ZOOM: {zoom_factor:.1f}x" if zoom_factor > 1.0 else ""
        red_status = " | 🔴 FIRE CONTROL ACTIVE" if red_crosshair_active else ""
        
        if red_crosshair_active:
            control_mode = "WASD=Servos(1°) | Arrows=Red Crosshair"
        else:
            control_mode = "WASD=1°"
            
        status_text = f"🎮 MANUAL CONTROL | {control_mode} | TAB=Toggle Fire Control | QE/ZX=Calibrate | SPACE=🔥FIRE{zoom_text}{red_status}"
        
        # Draw control status at bottom
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        status_x = 10
        status_y = frame_height - 20
        
        # Background for status
        cv2.rectangle(annotated_frame, (status_x - 5, status_y - 25), (status_x + text_size[0] + 10, status_y + 5), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (status_x - 5, status_y - 25), (status_x + text_size[0] + 10, status_y + 5), (0, 255, 0), 2)
        
        # Status text
        cv2.putText(annotated_frame, status_text, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # DRAW CENTER CROSSHAIR
        height, width = annotated_frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        bright_green = (0, 255, 0)  # BGR - pure green
        bright_white = (255, 255, 255)  # BGR - pure white
        
        # Draw crosshair
        crosshair_size = 50
        line_thickness = 6
        
        # Horizontal line
        cv2.line(annotated_frame, (center_x - crosshair_size, center_y), 
                (center_x + crosshair_size, center_y), bright_green, line_thickness)
        # Vertical line  
        cv2.line(annotated_frame, (center_x, center_y - crosshair_size), 
                (center_x, center_y + crosshair_size), bright_green, line_thickness)
        
        # Center dot
        cv2.circle(annotated_frame, (center_x, center_y), 15, bright_white, -1)
        cv2.circle(annotated_frame, (center_x, center_y), 15, bright_green, 4)
        
        # Label
        cv2.putText(annotated_frame, "CENTER", (center_x - 50, center_y - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, bright_green, 4)
        
        # DRAW RED CROSSHAIR (fire control solution)
        if red_crosshair_active and red_crosshair_x is not None and red_crosshair_y is not None:
            bright_red = (0, 0, 255)  # BGR - pure red
            red_size = 40
            red_thickness = 4
            
            # Draw red crosshair at calculated/adjusted position
            cv2.line(annotated_frame, (int(red_crosshair_x) - red_size, int(red_crosshair_y)), 
                    (int(red_crosshair_x) + red_size, int(red_crosshair_y)), bright_red, red_thickness)
            cv2.line(annotated_frame, (int(red_crosshair_x), int(red_crosshair_y) - red_size), 
                    (int(red_crosshair_x), int(red_crosshair_y) + red_size), bright_red, red_thickness)
            
            # Red center dot
            cv2.circle(annotated_frame, (int(red_crosshair_x), int(red_crosshair_y)), 8, bright_red, -1)
            
            # Label
            cv2.putText(annotated_frame, "FIRE CONTROL", (int(red_crosshair_x) - 70, int(red_crosshair_y) - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, bright_red, 3)
            

        
        # Prepare data for info panel
        info_data = {
            'drone_data': None,  # Will be filled by drone detection
            'log_messages': log_messages,
            'ai_debug': {'status': 'threaded_mode'}  # Show threaded mode status
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
            # Process key commands
            if key != 255 and key != 0:
                add_log(f"🔑 Key pressed: {key} (char: '{chr(key) if 32 <= key <= 126 else '?'}')")
            
            # TAB: Toggle red crosshair - create at center position and move servos
            if key == 9:  # TAB key
                if red_crosshair_active:
                    # Clear red crosshair if it's active
                    red_crosshair_active = False
                    red_crosshair_x = None
                    red_crosshair_y = None
                    add_log("🔴 RED CROSSHAIR CLEARED")
                else:
                    # Create red crosshair at AI calculated position (like before)
                    center_x = frame_width // 2
                    center_y = frame_height // 2
                    target_x = center_x
                    target_y = center_y
                    
                    # Try to get AI position if target detected
                    try:
                        if 'results' in locals() and results[0].boxes is not None and len(results[0].boxes) > 0:
                            ai_crosshair_x, ai_crosshair_y, ai_debug = get_ai_crosshair_position(
                                results[0], frame_width, frame_height, zoom_factor
                            )
                            # Use AI position if valid
                            if ai_debug.get('status') == 'ai_corrected':
                                target_x = ai_crosshair_x
                                target_y = ai_crosshair_y
                                add_log("🔴 LOCK ON: AI target detected")
                            else:
                                add_log("🔴 LOCK ON: Using center (no valid AI target)")
                        else:
                            add_log("🔴 LOCK ON: Using center (no detections)")
                    except Exception as e:
                        add_log(f"🔴 LOCK ON: Using center (AI error: {str(e)[:20]})")
                    
                    # Apply manual calibration offsets
                    global parallax_offset_x, parallax_offset_y
                    calibrated_x = target_x + parallax_offset_x
                    calibrated_y = target_y + parallax_offset_y
                    
                    # Keep within frame bounds
                    calibrated_x = max(0, min(calibrated_x, frame_width))
                    calibrated_y = max(0, min(calibrated_y, frame_height))
                    
                    # Activate red crosshair at AI position
                    red_crosshair_active = True
                    red_crosshair_x = calibrated_x
                    red_crosshair_y = calibrated_y
                    
                    # NOW move servos to put GREEN CENTER where TARGET OFFSET would be
                    # Calculate where target offset crosshair would be
                    fire_offset_x = calibrated_x - center_x
                    fire_offset_y = calibrated_y - center_y
                    target_offset_x = center_x - fire_offset_x
                    target_offset_y = center_y - fire_offset_y
                    
                    # Calculate how far to move center to reach target offset position
                    x_move_deviation = target_offset_x - center_x
                    y_move_deviation = target_offset_y - center_y
                    
                    # Convert to servo steps with fine-tuning factor
                    pixels_per_degree = frame_width / (config.CAMERA_FOV_HORIZONTAL / zoom_factor)
                    
                    # Add fine-tuning factor - adjust this if movement is too much/little
                    movement_scale = 0.8  # Reduce movement by 20% to account for calculation errors
                    
                    # Additional offset compensation for "3 inches up and right" error
                    # Convert 3 inches at typical distance to pixels (rough estimate)
                    distance_compensation_x = -65  # Move servo LEFT to compensate for rightward error (fine-tuned)
                    distance_compensation_y = 30   # Move servo DOWN to compensate for upward error (doubled)
                    
                    x_steps = (abs(x_move_deviation) / pixels_per_degree) * movement_scale
                    y_steps = (abs(y_move_deviation) / pixels_per_degree) * movement_scale
                    
                    # Apply distance compensation
                    x_compensation_steps = abs(distance_compensation_x) / pixels_per_degree
                    y_compensation_steps = abs(distance_compensation_y) / pixels_per_degree
                    
                    # Move servos to put center at target offset position
                    if x_steps > 1.0:  # Only move if significant deviation
                        if x_move_deviation > 0:  # Need to move center right, move servo LEFT (inverted for TAB)
                            direction = 'left'
                        else:  # Need to move center left, move servo RIGHT (inverted for TAB)
                            direction = 'right'
                        
                        step_size = min(x_steps + x_compensation_steps, 10.0)  # Add compensation
                        servo_cmd = {'type': 'manual', 'direction': direction, 'step': step_size}
                        try:
                            servo_command_queue.put_nowait(servo_cmd)
                            add_log(f"🎯 MOVING CENTER TO OFFSET X: {direction.upper()} {step_size:.1f}°")
                        except queue.Full:
                            add_log("❌ Servo queue full")
                    
                    if y_steps > 1.0:  # Only move if significant deviation  
                        if y_move_deviation > 0:  # Need to move center down, move servo down
                            direction = 'down'
                        else:  # Need to move center up, move servo up
                            direction = 'up'
                        
                        step_size = min(y_steps + y_compensation_steps, 10.0)  # Add compensation
                        servo_cmd = {'type': 'manual', 'direction': direction, 'step': step_size}
                        try:
                            servo_command_queue.put_nowait(servo_cmd)
                            add_log(f"🎯 MOVING CENTER TO OFFSET Y: {direction.upper()} {step_size:.1f}°")
                        except queue.Full:
                            add_log("❌ Servo queue full")
                    
                    add_log("🔴 RED FIRE CONTROL: AI calculated position")
                    add_log("   Servos moving to center target, then use arrows to fine-tune")

            # AI tracking disabled - too unstable
            elif key == ord('t') or key == ord('T'):
                add_log("❌ AI TRACKING DISABLED - Use manual control only")
            
            # Red crosshair movement (Arrow keys when red crosshair is active)
            if red_crosshair_active and red_crosshair_x is not None and red_crosshair_y is not None:
                if key in [82, 84, 81, 83]:  # Arrow keys only for crosshair movement
                    # Choose movement size for crosshair adjustment
                    move_step = 3  # pixels for crosshair fine adjustment
                    key_type = "ARROW"
                    
                    # Move red crosshair
                    if key == 82:  # Up arrow
                        red_crosshair_y = max(0, red_crosshair_y - move_step)
                        add_log(f"🔴 Red crosshair UP {move_step}px ({key_type})")
                    elif key == 84:  # Down arrow
                        red_crosshair_y = min(frame_height, red_crosshair_y + move_step)
                        add_log(f"🔴 Red crosshair DOWN {move_step}px ({key_type})")
                    elif key == 81:  # Left arrow
                        red_crosshair_x = max(0, red_crosshair_x - move_step)
                        add_log(f"🔴 Red crosshair LEFT {move_step}px ({key_type})")
                    elif key == 83:  # Right arrow
                        red_crosshair_x = min(frame_width, red_crosshair_x + move_step)
                        add_log(f"🔴 Red crosshair RIGHT {move_step}px ({key_type})")
            
            # SIMPLE: Just WASD for servo movement - KEEP IT SIMPLE!
            if key in [ord('w'), ord('s'), ord('a'), ord('d')]:
                current_step = 1  # Just use 1° - simple and works!
                key_type = "WASD"
                
                # Map keys to directions
                direction_map = {
                    ord('w'): 'up',     # W = up
                    ord('s'): 'down',   # S = down
                    ord('a'): 'right',  # A = servo right (camera pans left)
                    ord('d'): 'left'    # D = servo left (camera pans right)
                }
                
                direction = direction_map.get(key)
                if direction:
                    servo_cmd = {
                        'type': 'manual',
                        'direction': direction,
                        'step': current_step
                    }
                    try:
                        servo_command_queue.put_nowait(servo_cmd)
                        add_log(f"↗️ Manual {direction.upper()} {current_step}° ({key_type})")
                    except queue.Full:
                        add_log("❌ Servo queue full - try again")
            
            # System controls
            elif key == ord('c'):  # Center servos
                servo_cmd = {'type': 'center'}
                try:
                    servo_command_queue.put_nowait(servo_cmd)
                    add_log("🎯 Centering servos...")
                except queue.Full:
                    add_log("❌ Servo queue full - try again")
                    
            elif key == 32 or key == ord(' '):  # Fire blaster
                servo_cmd = {'type': 'fire'}
                try:
                    servo_command_queue.put_nowait(servo_cmd)
                    add_log("💥 FIRING BLASTER...")
                except queue.Full:
                    add_log("❌ Servo queue full - try again")
            
            # Zoom controls
            elif key == ord('+') or key == ord('='):
                zoom_factor = min(zoom_factor * config.ZOOM_STEP, config.MAX_ZOOM_FACTOR)
                add_log(f"🔍 Zoom IN: {zoom_factor:.1f}x")
                
            elif key == ord('-') or key == ord('_'):
                zoom_factor = max(zoom_factor / config.ZOOM_STEP, config.DEFAULT_ZOOM_FACTOR)
                if zoom_factor == config.DEFAULT_ZOOM_FACTOR:
                    add_log("🔍 Zoom RESET: 1.0x (no zoom)")
                else:
                    add_log(f"🔍 Zoom OUT: {zoom_factor:.1f}x")
            
            elif key == ord('0'):
                zoom_factor = config.DEFAULT_ZOOM_FACTOR
                add_log("🔍 ZOOM RESET: 1.0x")
                
            # Parallax calibration controls (larger steps for significant corrections)
            elif key == ord('q'):  # Move fire control LEFT
                parallax_offset_x -= 25
                add_log(f"🎯 Parallax X offset: {parallax_offset_x} (LEFT)")
                
            elif key == ord('e'):  # Move fire control RIGHT  
                parallax_offset_x += 25
                add_log(f"🎯 Parallax X offset: {parallax_offset_x} (RIGHT)")
                
            elif key == ord('z'):  # Move fire control UP
                parallax_offset_y -= 25
                add_log(f"🎯 Parallax Y offset: {parallax_offset_y} (UP)")
                
            elif key == ord('x'):  # Move fire control DOWN
                parallax_offset_y += 25
                add_log(f"🎯 Parallax Y offset: {parallax_offset_y} (DOWN)")
                
            elif key == ord('r') or key == ord('R'):  # Reset parallax offsets
                parallax_offset_x = 0
                parallax_offset_y = 0
                add_log("🎯 Parallax offsets RESET to (0, 0)")
                
            elif key == ord('h'):  # Help
                print("\n" + "="*60)
                print("🎮 THREADED DRONE TRACKER CONTROLS:")
                print("="*60)
                print("🤖 AI TRACKING:")
                print("   T - Toggle AI auto-tracking ON/OFF")
                print("   (AI runs in separate thread for max performance)")
                print("")
                print("🕹️ MANUAL MOVEMENT:")
                print("   WASD - Move servos (1°)")
                print("   ↑/W - Tilt UP    ↓/S - Tilt DOWN")
                print("   ←/A - Pan LEFT   →/D - Pan RIGHT")
                print("")
                print("🔍 DIGITAL ZOOM:")
                print("   + - Zoom IN (up to 5x)")
                print("   - - Zoom OUT")
                print("   0 - Reset zoom to 1x")
                print("")
                print("⚡ FIRE CONTROL SYSTEM:")
                print("   1. Move camera with WASD to get target in green center")
                print("   2. Press TAB to create RED fire control crosshair")
                print("   3. Use ARROW KEYS to fine-tune red crosshair on target")
                print("   4. Press SPACEBAR to FIRE at red crosshair position")
                print("   5. TAB again to clear and start over")
                print("")
                print("🎯 PARALLAX CALIBRATION:")
                print("   Q/E - Adjust fire control LEFT/RIGHT")
                print("   Z/X - Adjust fire control UP/DOWN") 
                print("   R - Reset calibration to center")
                print("")
                print("⚡ SYSTEM:")
                print("   C - Center servos")
                print("   H - Show this help")
                print("   Q - Quit")
                print("="*60)
                print("🚀 PERFORMANCE: All threads run independently!")
                print("🤖 AI tracking: Dedicated thread (max responsiveness)")
                print("🎯 Servo control: Dedicated thread (20Hz update rate)")
                print("🖥️ GUI: Dedicated thread (smooth display)")
                print("📹 Camera/YOLO: Main thread (max detection speed)")
                print("="*60)
        
        # Minimal sleep to prevent CPU overload - reduced for max performance
        time.sleep(0.001)  # 1ms - allows ~1000 FPS processing
        
        frame_count += 1

    # Cleanup - shutdown all threads
    print("🛑 Shutting down all threads...")
    ai_tracking_thread_running = False
    
    # Send shutdown signals
    try:
        ai_tracking_queue.put_nowait(None)
        servo_command_queue.put_nowait(None)
    except:
        pass
    
    # Wait for threads to finish
    ai_thread.join(timeout=1.0)
    servo_thread.join(timeout=1.0)
    gui_thread_handle.join(timeout=1.0)
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ All threads stopped - goodbye!")

if __name__ == "__main__":
    main()