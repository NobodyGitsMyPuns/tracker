#!/usr/bin/env python3
"""
🚁 ESP32 Drone Integration Module
Connects YOLO camera detection to ESP32 servo tracking
"""

import requests
import time
import json
from typing import Optional, Tuple
from config import config

class ESP32DroneTracker:
    def __init__(self, esp32_ip: str = None, timeout: float = None):
        """
        Initialize ESP32 drone tracker connection
        
        Args:
            esp32_ip: IP address of ESP32 servo controller (defaults to config)
            timeout: HTTP request timeout in seconds (defaults to config)
        """
        self.esp32_ip = esp32_ip or config.ESP32_IP
        self.base_url = f"http://{self.esp32_ip}"
        self.timeout = timeout or config.ESP32_TIMEOUT
        self.last_command_time = 0
        self.command_rate_limit = 0.1   # 10 commands/sec for 4090 speed!
        self.tracking_active = False
        
        print(f"🚁 ESP32 Drone Tracker initialized: {self.base_url}")
        
        # Test connection
        if self.test_connection():
            print("✅ ESP32 connection successful!")
            self.start_sweep_mode()
        else:
            print("❌ ESP32 connection failed - check IP and network")
    
    def test_connection(self) -> bool:
        """Test if ESP32 is reachable"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return response.status_code == 200
        except:
            return False
    
    def send_drone_position(self, x_offset: float, y_offset: float) -> bool:
        """
        Send drone position to ESP32 for tracking with retry logic
        
        Args:
            x_offset: Horizontal offset in pixels from camera center (- = left, + = right)
            y_offset: Vertical offset in pixels from camera center (- = up, + = down)
            
        Returns:
            True if command sent successfully
        """
        # Rate limiting to prevent overwhelming ESP32
        current_time = time.time()
        if current_time - self.last_command_time < self.command_rate_limit:
            return False
        
        # Retry logic for intermittent connections (reduced retries)
        max_retries = 1  # Only 1 retry to avoid overwhelming ESP32
        for attempt in range(max_retries):
            try:
                data = {
                    "x_offset": int(x_offset),
                    "y_offset": int(y_offset)
                }
                
                response = requests.post(
                    f"{self.base_url}/track",
                    json=data,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.tracking_active = True
                    self.last_command_time = current_time
                    
                    print(f"🎯 ESP32 TRACKING: X={x_offset:+.0f}px, Y={y_offset:+.0f}px → "
                          f"Pan={result.get('pan', '?')}°, Tilt={result.get('tilt', '?')}°")
                    return True
                else:
                    print(f"❌ ESP32 HTTP error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Brief delay before retry
                        continue
                    
            except requests.exceptions.Timeout:
                print(f"⚠️ ESP32 timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.05)  # Brief delay before retry
                    continue
            except requests.exceptions.ConnectionError:
                print(f"⚠️ ESP32 connection lost (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Brief delay before retry
                    continue
            except Exception as e:
                print(f"⚠️ ESP32 error: {str(e)[:30]}")
                break
        
        return False
    
    def center_servos(self) -> bool:
        """Center both servos with retry logic"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.base_url}/center", timeout=self.timeout)
                if response.status_code == 200:
                    print("🎯 ESP32 servos centered")
                    self.tracking_active = False
                    return True
            except requests.exceptions.Timeout:
                print(f"⚠️ Center timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    continue
            except Exception as e:
                print(f"⚠️ Center error: {str(e)[:30]}")
                break
        return False
    
    def start_sweep_mode(self) -> bool:
        """Start sweep mode (search for drones)"""
        try:
            response = requests.get(f"{self.base_url}/sweep", timeout=self.timeout)
            if response.status_code == 200:
                print("🔄 ESP32 sweep mode started")
                self.tracking_active = False
                return True
        except Exception as e:
            print(f"⚠️ Failed to start ESP32 sweep: {e}")
        return False
    
    def stop_movement(self) -> bool:
        """Stop all servo movement"""
        try:
            response = requests.get(f"{self.base_url}/stop", timeout=self.timeout)
            if response.status_code == 200:
                print("⏸️ ESP32 movement stopped")
                self.tracking_active = False
                return True
        except Exception as e:
            print(f"⚠️ Failed to stop ESP32 movement: {e}")
        return False
    
    def get_status(self) -> Optional[dict]:
        """Get current ESP32 status"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️ Failed to get ESP32 status: {e}")
        return None


def add_esp32_tracking_to_yolo():
    """
    Integration instructions for adding ESP32 tracking to yolo_camera_detection.py
    """
    
    integration_code = '''
# Add this import at the top of yolo_camera_detection.py:
from esp32_drone_integration import ESP32DroneTracker

# Add this after line 210 (after frame_height definition):
# Initialize ESP32 drone tracker
esp32_tracker = ESP32DroneTracker("10.0.0.70")  # Use your ESP32's IP

# Add this code inside the main detection loop, around line 305 after bullseye drawing:
# Send drone position to ESP32 for tracking
if 'drone_position_data' in locals() and esp32_tracker:
    data = drone_position_data
    esp32_tracker.send_drone_position(
        data['deviation_x'],  # X offset in pixels
        data['deviation_y']   # Y offset in pixels
    )

# Add this keyboard control around line 580 (in the key handling section):
elif key == ord('c'):  # 'c' key to center servos
    if esp32_tracker:
        esp32_tracker.center_servos()
elif key == ord('s'):  # 's' key to start sweep mode
    if esp32_tracker:
        esp32_tracker.start_sweep_mode()
elif key == ord('x'):  # 'x' key to stop movement
    if esp32_tracker:
        esp32_tracker.stop_movement()
'''
    
    print("🔧 ESP32 Integration Code for YOLO:")
    print("=" * 50)
    print(integration_code)
    print("=" * 50)
    print("\n💡 Instructions:")
    print("1. Make sure ESP32 is running and connected to network")
    print("2. Update ESP32 IP address in the code above")
    print("3. Add the integration code to yolo_camera_detection.py")
    print("4. Run your YOLO detection - it will automatically track drones!")
    print("\n🎮 Keyboard Controls:")
    print("   'c' - Center servos")
    print("   's' - Start sweep mode")
    print("   'x' - Stop movement")
    print("   'q' - Quit")


if __name__ == "__main__":
    print("🚁 ESP32 Drone Integration Test")
    print("=" * 40)
    
    # Test ESP32 connection
    tracker = ESP32DroneTracker("10.0.0.70")  # Update with your ESP32 IP
    
    if tracker.test_connection():
        print("\n🧪 Running test sequence...")
        
        # Test center
        tracker.center_servos()
        time.sleep(2)
        
        # Test tracking commands
        test_positions = [
            (-100, -50, "Drone left and up"),
            (150, 75, "Drone right and down"), 
            (0, 0, "Drone centered"),
            (-50, 100, "Drone left and down"),
            (0, 0, "Back to center")
        ]
        
        for x, y, desc in test_positions:
            print(f"🎯 Testing: {desc}")
            tracker.send_drone_position(x, y)
            time.sleep(3)
        
        # Return to sweep mode
        tracker.start_sweep_mode()
        print("✅ Test complete!")
    else:
        print("❌ Cannot connect to ESP32")
        print("💡 Make sure ESP32 is powered and on the same network")
    
    # Show integration instructions
    add_esp32_tracking_to_yolo()
