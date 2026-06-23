#!/usr/bin/env python3
"""
ESP32 Drone Integration Module
Connects YOLO camera detection to ESP32 servo tracking
"""

import requests
import time
import json
from typing import List, Optional, Tuple
from config import config


# Servo travel limits enforced by the ESP32 firmware (ESP32_OTA_Servo.ino):
# pan is constrained to 0-180 and tilt to 20-160. Mirror them on the client so
# an out-of-range command is never sent over the wire and the servos are never
# driven against their mechanical stops (the firmware would clamp to exactly
# these values anyway, so the physical result is unchanged).
PAN_ANGLE_MIN, PAN_ANGLE_MAX = 0, 180
TILT_ANGLE_MIN, TILT_ANGLE_MAX = 20, 160


def clamp_servo_angles(pan_angle: int, tilt_angle: int) -> Tuple[int, int]:
    """Clamp a requested (pan, tilt) to the servos' safe travel.

    Pan is clamped to ``[PAN_ANGLE_MIN, PAN_ANGLE_MAX]`` and tilt to
    ``[TILT_ANGLE_MIN, TILT_ANGLE_MAX]`` — the same ranges the firmware enforces
    — so a caller that computes an out-of-range angle can't push the hardware
    past its limits. In-range values pass through unchanged.
    """
    safe_pan = max(PAN_ANGLE_MIN, min(PAN_ANGLE_MAX, pan_angle))
    safe_tilt = max(TILT_ANGLE_MIN, min(TILT_ANGLE_MAX, tilt_angle))
    return safe_pan, safe_tilt


def plan_servo_moves(angle_x: float, angle_y: float,
                     move_threshold: float = 5,
                     max_step: int = 10) -> List[Tuple[str, int]]:
    """Translate per-axis angular error into the servo step commands to send.

    Each axis is corrected **independently** and proportionally to its *own*
    angular error (in degrees), capped at ``max_step`` degrees:

    - pan moves ``right`` when ``angle_x > 0`` else ``left``;
    - tilt moves ``down`` when ``angle_y > 0`` else ``up``.

    An axis only contributes a command when its error exceeds ``move_threshold``,
    so a purely-horizontal error no longer drags a (bogus) tilt command along,
    and vice-versa. Returns a list of ``(direction, step)`` tuples in pan-then-
    tilt order; the list is empty when the target is within ``move_threshold``
    on both axes.
    """
    moves: List[Tuple[str, int]] = []
    if abs(angle_x) > move_threshold:
        step = int(min(abs(angle_x), max_step))
        if step > 0:
            moves.append(("right" if angle_x > 0 else "left", step))
    if abs(angle_y) > move_threshold:
        step = int(min(abs(angle_y), max_step))
        if step > 0:
            moves.append(("down" if angle_y > 0 else "up", step))
    return moves


class ESP32DroneTracker:
    def __init__(self, esp32_ip: str = None, timeout: float = None,
                 command_rate_limit: float = None):
        """
        Initialize ESP32 drone tracker connection

        Args:
            esp32_ip: IP address of ESP32 servo controller (defaults to config)
            timeout: HTTP request timeout in seconds (defaults to config)
            command_rate_limit: Minimum seconds between servo commands
                (defaults to config.COMMAND_RATE_LIMIT)
        """
        self.esp32_ip = esp32_ip or config.ESP32_IP
        self.base_url = f"http://{self.esp32_ip}"
        self.timeout = timeout or config.ESP32_TIMEOUT
        self.last_command_time = 0
        # Respect the configurable COMMAND_RATE_LIMIT (.env) instead of a
        # hardcoded value; default is 0.1s == 10 commands/sec.
        self.command_rate_limit = command_rate_limit or config.COMMAND_RATE_LIMIT
        self.tracking_active = False

        print(f"ESP32 Drone Tracker initialized: {self.base_url}")

    def start_sweep_mode(self) -> bool:
        """Start automatic sweep mode"""
        try:
            response = requests.get(f"{self.base_url}/sweep", timeout=self.timeout)
            if response.status_code == 200:
                print("SUCCESS: Sweep mode started")
                return True
            else:
                print(f"ERROR: Sweep mode failed - HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: Sweep mode failed - {str(e)}")
            return False

    def stop_sweep_mode(self) -> bool:
        """Stop sweep mode"""
        try:
            response = requests.get(f"{self.base_url}/stop", timeout=self.timeout)
            if response.status_code == 200:
                print("SUCCESS: Sweep mode stopped")
                return True
            else:
                print(f"ERROR: Stop sweep failed - HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: Stop sweep failed - {str(e)}")
            return False

    def move_to_position(self, pan_angle: int, tilt_angle: int) -> bool:
        """
        Move servos to specific position

        The requested angles are clamped to the servos' safe travel before the
        command is sent — pan to 0-180 and tilt to 20-160 degrees — matching the
        limits the ESP32 firmware enforces. An out-of-range request is clamped
        (and logged) instead of being driven against the mechanical stops.

        Args:
            pan_angle: Pan angle in degrees (clamped to 0-180)
            tilt_angle: Tilt angle in degrees (clamped to 20-160)
        """
        try:
            safe_pan, safe_tilt = clamp_servo_angles(pan_angle, tilt_angle)
            if (safe_pan, safe_tilt) != (pan_angle, tilt_angle):
                print(
                    f"WARNING: Requested position pan={pan_angle}, "
                    f"tilt={tilt_angle} out of range; clamped to "
                    f"pan={safe_pan}, tilt={safe_tilt}"
                )

            # Rate limiting to prevent servo overload
            current_time = time.time()
            if current_time - self.last_command_time < self.command_rate_limit:
                time.sleep(self.command_rate_limit - (current_time - self.last_command_time))

            url = f"{self.base_url}/move?pan={safe_pan}&tilt={safe_tilt}"
            response = requests.get(url, timeout=self.timeout)
            self.last_command_time = time.time()

            if response.status_code == 200:
                print(f"SUCCESS: Moved to pan={safe_pan}, tilt={safe_tilt}")
                return True
            else:
                print(f"ERROR: Move failed - HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: Move failed - {str(e)}")
            return False

    def track_drone(self, x_deviation: float, y_deviation: float,
                   frame_width: int, frame_height: int,
                   camera_fov_horizontal: float = 60, camera_fov_vertical: float = 45) -> bool:
        """
        Track detected drone by moving servos

        Args:
            x_deviation: Horizontal pixel deviation from center
            y_deviation: Vertical pixel deviation from center
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            camera_fov_horizontal: Camera horizontal field of view in degrees
            camera_fov_vertical: Camera vertical field of view in degrees
        """
        try:
            # Convert pixel deviation to angle
            angle_per_pixel_x = camera_fov_horizontal / frame_width
            angle_per_pixel_y = camera_fov_vertical / frame_height

            angle_x = x_deviation * angle_per_pixel_x
            angle_y = y_deviation * angle_per_pixel_y

            # Plan per-axis corrections (each axis sized to its own error).
            moves = plan_servo_moves(angle_x, angle_y, move_threshold=5, max_step=10)
            if not moves:
                # Target is centered, no movement needed
                return True

            # Rate limiting
            current_time = time.time()
            if current_time - self.last_command_time < self.command_rate_limit:
                return False

            for direction, step in moves:
                url = f"{self.base_url}/{direction}?step={step}"
                response = requests.get(url, timeout=self.timeout)
                if response.status_code != 200:
                    print(f"ERROR: Tracking failed - HTTP {response.status_code}")
                    return False

            self.last_command_time = time.time()
            self.tracking_active = True
            moved = ", ".join(f"{direction} {step}°" for direction, step in moves)
            print(f"SUCCESS: Tracking - moved {moved}")
            return True

        except Exception as e:
            print(f"ERROR: Tracking failed - {str(e)}")
            return False

    def center_servos(self) -> bool:
        """Center both servos to home position"""
        try:
            response = requests.get(f"{self.base_url}/center", timeout=self.timeout)
            if response.status_code == 200:
                print("SUCCESS: Servos centered")
                self.tracking_active = False
                return True
            else:
                print(f"ERROR: Center failed - HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: Center failed - {str(e)}")
            return False

    def fire_blaster(self, duration_ms: int = 150) -> bool:
        """Fire the blaster for specified duration"""
        try:
            url = f"{self.base_url}/fire?duration={duration_ms}"
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                print(f"SUCCESS: Blaster fired for {duration_ms}ms")
                return True
            else:
                print(f"ERROR: Fire failed - HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: Fire failed - {str(e)}")
            return False

    def stop_movement(self) -> bool:
        """Stop all servo movement"""
        try:
            response = requests.get(f"{self.base_url}/stop", timeout=self.timeout)
            if response.status_code == 200:
                print("SUCCESS: Movement stopped")
                self.tracking_active = False
                return True
            else:
                print(f"ERROR: Stop failed - HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: Stop failed - {str(e)}")
            return False

    def test_connection(self) -> bool:
        """Test connection to ESP32"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                print("SUCCESS: ESP32 connection test passed")
                return True
            else:
                print(f"WARNING: ESP32 responded with HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"ERROR: ESP32 connection test failed - {str(e)}")
            return False

    def get_status(self) -> Optional[dict]:
        """Get current servo positions and status"""
        try:
            response = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"ERROR: Status request failed - HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"ERROR: Status request failed - {str(e)}")
            return None

if __name__ == "__main__":
    # Test the ESP32 connection
    tracker = ESP32DroneTracker()

    print("Testing ESP32 connection...")
    if tracker.test_connection():
        print("SUCCESS: ESP32 is responding")

        print("Testing servo centering...")
        tracker.center_servos()

        print("Testing movement...")
        tracker.move_to_position(120, 60)
        time.sleep(1)
        tracker.center_servos()

    else:
        print("ERROR: ESP32 not responding - check network connection")
