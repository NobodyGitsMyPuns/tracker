#!/usr/bin/env python3
"""
Update configuration for 4K camera and centered crosshair
"""

import os

def create_4k_env():
    """Create .env file optimized for 4K camera with centered crosshair"""
    
    env_content = """# Drone Tracker Configuration
# Updated for 4K camera and centered crosshair

# =============================================================================
# WiFi Configuration
# ============================================================================="
WIFI_SSID=
WIFI_PASSWORD=

# ESP32 Hardware Configuration
ESP32_IP=10.0.0.70
ESP32_TIMEOUT=0.5

# Camera Configuration - 4K RESOLUTION
CAMERA_INDEX=0
CAMERA_WIDTH=3840
CAMERA_HEIGHT=2160

# Servo Configuration (degrees)
PAN_HOME=102
TILT_HOME=90
CENTER_ANGLE=90

# AI Detection Configuration
CONFIDENCE_THRESHOLD=0.05
IOU_THRESHOLD=0.2
MAX_DETECTIONS=200

# Camera Field of View (degrees)
CAMERA_FOV_HORIZONTAL=60
CAMERA_FOV_VERTICAL=45

# Movement Configuration (degrees per keypress)
MANUAL_STEP_SIZE=5
MANUAL_STEP_SIZE_SHIFT=15

# Crosshair Configuration - CENTERED
CROSSHAIR_OFFSET_X=0.0
CROSSHAIR_OFFSET_Y=0.0

# UI/Display Configuration - 4K OPTIMIZED
MAIN_WINDOW_WIDTH=1920
MAIN_WINDOW_HEIGHT=1080
MAIN_WINDOW_POS_X=50
MAIN_WINDOW_POS_Y=50
INFO_PANEL_WIDTH=600
INFO_PANEL_HEIGHT=400
INFO_PANEL_POS_X=2000
INFO_PANEL_POS_Y=50

# Digital Zoom Configuration
DEFAULT_ZOOM_FACTOR=1.0
MAX_ZOOM_FACTOR=5.0
ZOOM_STEP=1.2

# Detection/Tracking Configuration
MOVEMENT_THRESHOLD=25
MAX_TRACKING_HISTORY=5
DEBUG_FRAME_INTERVAL=60

# Logging Configuration
MAX_LOG_MESSAGES=8
LOG_MESSAGE_MAX_LENGTH=70

# Network Configuration
HTTP_TIMEOUT=1.0
HTTP_RETRIES=1
COMMAND_RATE_LIMIT=0.1
"""
    
    # Write the .env file
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("SUCCESS: Updated .env file with 4K camera settings!")
    print("CAMERA: Resolution set to 3840x2160 (4K)")
    print("CROSSHAIR: Centered (0,0 offset)")
    print("DISPLAY: Window set to 1920x1080")
    print("TRACKING: Movement threshold increased for 4K")

if __name__ == "__main__":
    create_4k_env()
