# 🚁 AI Drone Tracker System

A real-time drone detection and tracking system using YOLO AI and ESP32 servo control.

## 🎯 Core Files

### Main System
- **`yolo_camera_detection.py`** - Main AI detection system with threaded GUI
- **`esp32_drone_integration.py`** - Python-ESP32 communication layer
- **`ESP32_OTA_Servo/ESP32_OTA_Servo.ino`** - ESP32 servo controller firmware

### Dependencies & Models
- **`requirements.txt`** - Python package dependencies
- **`yolov8*.pt`** - YOLO AI model files (nano, small, large, extra-large)
- **`upload_esp32_ota.bat`** - ESP32 OTA update script

## 🚀 Quick Start

### 1. Configuration Setup
```bash
# Copy template and edit with your settings
copy env.sample .env
# OR run the setup script to create .env with current values
python setup_env.py
# Edit .env with your WiFi credentials and preferences
```

### 2. Python Setup
```bash
pip install -r requirements.txt
python generate_arduino_config.py  # Generate Arduino config from .env
python yolo_camera_detection.py
```

### 3. ESP32 Setup
1. Upload `ESP32_OTA_Servo/ESP32_OTA_Servo.ino` to your ESP32
2. WiFi credentials are auto-loaded from generated `config.h`
3. Connect servos to pins 18 (pan) and 19 (tilt)

### 4. Configuration Files
- **`.env`** - Main configuration (WiFi, detection, UI, servo settings)
- **`env.sample`** - Complete template with all available options
- **`config.py`** - Python configuration loader with defaults
- **`setup_env.py`** - Script to create .env with current hardcoded values
- **`generate_arduino_config.py`** - Generates Arduino config from .env
- **`ESP32_OTA_Servo/config.h`** - Auto-generated Arduino config (don't edit)

## 🎮 Controls

### Movement
- **WASD** - Fine movement (5°)
- **Arrow Keys** - Coarse movement (15°)
- **C** - Center servos
- **X** - Stop movement

### Digital Zoom
- **+** - Zoom in (up to 5x)
- **-** - Zoom out
- **0** - Reset zoom

### System
- **SPACEBAR** - Fire blaster (if connected)
- **H** - Show help
- **Q** - Quit

## 📊 Features

- **Real-time YOLO detection** with GPU acceleration
- **Threaded GUI** for responsive camera feed
- **Digital zoom** with keyboard controls
- **Separate info panel** for logs and targeting data
- **ESP32 servo control** with OTA updates
- **Drone-optimized detection** for aerial objects
- **Movement tracking** for confirmed drone identification

## 📁 Archive

Old code, tests, and documentation are organized in the `archive/` folder:
- `archive/old_arduino_projects/` - Previous ESP32/Arduino experiments
- `archive/old_python_scripts/` - Old detection scripts and tests
- `archive/documentation/` - Setup guides and configuration files
- `archive/environments/` - Old Python virtual environments

## ⚙️ Configuration

All settings are managed through the `.env` file. **Complete list of 25+ configurable options:**

```env
# Core Settings (Required)
WIFI_SSID=your_network_name
WIFI_PASSWORD=your_network_password
ESP32_IP=10.0.0.70

# Servo Positions
PAN_HOME=102               # 12° right of center
TILT_HOME=90               # Center position

# Detection Settings
CONFIDENCE_THRESHOLD=0.05  # Lower = more sensitive
CAMERA_FOV_HORIZONTAL=60   # Camera field of view

# UI/Display (25+ options available)
MAIN_WINDOW_WIDTH=1280     # Window size
CROSSHAIR_OFFSET_X=10.0    # Crosshair position
ZOOM_STEP=1.2              # Zoom increment
MAX_LOG_MESSAGES=8         # Log display count
# ...and many more in env.sample
```

**Configuration Flow:**
1. Edit `.env` with your settings
2. Run `python generate_arduino_config.py` 
3. Upload Arduino code (includes generated config)
4. Run `python yolo_camera_detection.py`

## 🔧 Hardware Requirements

- **Camera** - USB webcam or built-in camera
- **ESP32** development board
- **Servos** - 2x servo motors for pan/tilt
- **Optional** - Relay module for blaster control

## 💡 Tips

- Use YOLOv8x or YOLOv10x models for best accuracy
- Ensure ESP32 and computer are on same network
- Adjust `PAN_HOME` and `TILT_HOME` for your servo mounting
- Use digital zoom for detailed tracking of distant targets

---

*System optimized for RTX 4090 with CUDA acceleration*
