#!/usr/bin/env python3
"""
 Environment Setup Script
Creates .env file with current hardcoded values
"""

import os

def setup_env_file():
 """Create .env file with extracted values from old hardcoded settings"""

 # Values extracted from the old hardcoded settings
 env_content = """# Drone Tracker Configuration
# Generated from setup_env.py - edit as needed

# =============================================================================
# WiFi Configuration (UPDATE THESE!)
# =============================================================================
WIFI_SSID=
WIFI_PASSWORD=

# =============================================================================
# ESP32 Hardware Configuration
# =============================================================================
ESP32_IP=10.0.0.70
ESP32_TIMEOUT=0.5

# =============================================================================
# Camera Configuration
# =============================================================================
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

# =============================================================================
# Servo Configuration (degrees)
# =============================================================================
PAN_HOME=102 # 12 degrees right from center
TILT_HOME=90 # Center position (was 45 in some versions)
CENTER_ANGLE=90

# =============================================================================
# AI Detection Configuration
# =============================================================================
CONFIDENCE_THRESHOLD=0.05
IOU_THRESHOLD=0.2
MAX_DETECTIONS=200

# =============================================================================
# Camera Field of View (degrees)
# =============================================================================
CAMERA_FOV_HORIZONTAL=60
CAMERA_FOV_VERTICAL=45

# =============================================================================
# Movement Configuration (degrees per keypress)
# =============================================================================
MANUAL_STEP_SIZE=5
MANUAL_STEP_SIZE_SHIFT=15
"""

 if os.path.exists('.env'):
 response = input("WARNING: .env file already exists. Overwrite? (y/N): ")
 if response.lower() != 'y':
 print("ERROR: Cancelled - .env file not modified")
 return

 with open('.env', 'w') as f:
 f.write(env_content)

 print("SUCCESS: Created .env file with current settings")
 print(" Please review and update WiFi credentials in .env file")
 print(" .env file is excluded from git for security")

if __name__ == "__main__":
 setup_env_file()
