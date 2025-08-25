#!/usr/bin/env python3
"""
Drone Tracker Configuration
Centralized configuration for the drone tracking system
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for drone tracker system"""
    
    # WiFi Configuration
    WIFI_SSID = os.getenv('WIFI_SSID', 'your_wifi_name')
    WIFI_PASSWORD = os.getenv('WIFI_PASSWORD', 'your_wifi_password')
    
    # ESP32 Configuration
    ESP32_IP = os.getenv('ESP32_IP', '10.0.0.70')
    ESP32_TIMEOUT = float(os.getenv('ESP32_TIMEOUT', '0.5'))
    
    # Camera Configuration
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '3840'))
    CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '2160'))
    
    # Servo Configuration (degrees)
    PAN_HOME = int(os.getenv('PAN_HOME', '102'))
    TILT_HOME = int(os.getenv('TILT_HOME', '90'))
    CENTER_ANGLE = int(os.getenv('CENTER_ANGLE', '90'))
    
    # Detection Configuration
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.05'))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.2'))
    MAX_DETECTIONS = int(os.getenv('MAX_DETECTIONS', '200'))
    
    # Field of View (degrees)
    CAMERA_FOV_HORIZONTAL = int(os.getenv('CAMERA_FOV_HORIZONTAL', '60'))
    CAMERA_FOV_VERTICAL = int(os.getenv('CAMERA_FOV_VERTICAL', '45'))
    
    # Movement Configuration
    MANUAL_STEP_SIZE = int(os.getenv('MANUAL_STEP_SIZE', '5'))
    MANUAL_STEP_SIZE_SHIFT = int(os.getenv('MANUAL_STEP_SIZE_SHIFT', '15'))
    
    # UI/Display Configuration
    MAIN_WINDOW_WIDTH = int(os.getenv('MAIN_WINDOW_WIDTH', '1920'))
    MAIN_WINDOW_HEIGHT = int(os.getenv('MAIN_WINDOW_HEIGHT', '1080'))
    MAIN_WINDOW_NAME = os.getenv('MAIN_WINDOW_NAME', 'YOLO Drone Tracker - HD')
    MAIN_WINDOW_POS_X = int(os.getenv('MAIN_WINDOW_POS_X', '50'))
    MAIN_WINDOW_POS_Y = int(os.getenv('MAIN_WINDOW_POS_Y', '50'))
    
    # Info Panel Settings
    INFO_PANEL_WIDTH = int(os.getenv('INFO_PANEL_WIDTH', '600'))
    INFO_PANEL_HEIGHT = int(os.getenv('INFO_PANEL_HEIGHT', '400'))
    INFO_PANEL_NAME = os.getenv('INFO_PANEL_NAME', 'System Info & Logs')
    INFO_PANEL_POS_X = int(os.getenv('INFO_PANEL_POS_X', '2000'))
    INFO_PANEL_POS_Y = int(os.getenv('INFO_PANEL_POS_Y', '50'))
    
    # Queue Sizes
    DISPLAY_QUEUE_SIZE = int(os.getenv('DISPLAY_QUEUE_SIZE', '2'))
    INFO_QUEUE_SIZE = int(os.getenv('INFO_QUEUE_SIZE', '2'))
    
    # Digital Zoom Settings
    DEFAULT_ZOOM_FACTOR = float(os.getenv('DEFAULT_ZOOM_FACTOR', '1.0'))
    MAX_ZOOM_FACTOR = float(os.getenv('MAX_ZOOM_FACTOR', '5.0'))
    ZOOM_STEP = float(os.getenv('ZOOM_STEP', '1.2'))
    
    # Crosshair Offset (degrees from center) - NOW CENTERED
    CROSSHAIR_OFFSET_X = float(os.getenv('CROSSHAIR_OFFSET_X', '0.0'))  # CENTERED
    CROSSHAIR_OFFSET_Y = float(os.getenv('CROSSHAIR_OFFSET_Y', '0.0'))  # CENTERED
    
    # Colors (BGR format for OpenCV)
    COLOR_GREEN = (0, 255, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_BLUE = (255, 0, 0)
    
    # Text Settings
    FONT_SCALE_SMALL = float(os.getenv('FONT_SCALE_SMALL', '0.4'))
    FONT_SCALE_MEDIUM = float(os.getenv('FONT_SCALE_MEDIUM', '0.5'))
    FONT_SCALE_LARGE = float(os.getenv('FONT_SCALE_LARGE', '0.7'))
    
    # Logging Settings
    MAX_LOG_MESSAGES = int(os.getenv('MAX_LOG_MESSAGES', '8'))
    LOG_MESSAGE_MAX_LENGTH = int(os.getenv('LOG_MESSAGE_MAX_LENGTH', '70'))
    
    # Detection/Tracking Configuration
    MOVEMENT_THRESHOLD = int(os.getenv('MOVEMENT_THRESHOLD', '25'))  # Increased for 4K
    MAX_TRACKING_HISTORY = int(os.getenv('MAX_TRACKING_HISTORY', '5'))
    
    # Drone Classes (objects YOLO might classify drones as)
    DRONE_CLASSES = {'clock', 'bird', 'kite', 'frisbee', 'sports ball', 
                    'donut', 'apple', 'orange', 'cell phone', 'remote'}
    
    # Frame Processing
    DEBUG_FRAME_INTERVAL = int(os.getenv('DEBUG_FRAME_INTERVAL', '60'))
    
    # Network/Communication Configuration
    HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '1.0'))
    HTTP_RETRIES = int(os.getenv('HTTP_RETRIES', '1'))
    COMMAND_RATE_LIMIT = float(os.getenv('COMMAND_RATE_LIMIT', '0.1'))
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("Current Configuration:")
        print(f"   WiFi SSID: {cls.WIFI_SSID}")
        print(f"   WiFi Password: {'*' * len(cls.WIFI_PASSWORD)}")
        print(f"   ESP32 IP: {cls.ESP32_IP}")
        print(f"   ESP32 Timeout: {cls.ESP32_TIMEOUT}s")
        print(f"   Camera: {cls.CAMERA_INDEX} ({cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT})")
        print(f"   Servo Home: Pan={cls.PAN_HOME}°, Tilt={cls.TILT_HOME}°")
        print(f"   Detection: conf={cls.CONFIDENCE_THRESHOLD}, iou={cls.IOU_THRESHOLD}")
        print(f"   FOV: {cls.CAMERA_FOV_HORIZONTAL}°x{cls.CAMERA_FOV_VERTICAL}°")
        print(f"   Movement: {cls.MANUAL_STEP_SIZE}°/{cls.MANUAL_STEP_SIZE_SHIFT}°")
        print(f"   Crosshair: {cls.CROSSHAIR_OFFSET_X}°x, {cls.CROSSHAIR_OFFSET_Y}°y (CENTERED)")

# Create a default config instance
config = Config()

if __name__ == "__main__":
    config.print_config()
