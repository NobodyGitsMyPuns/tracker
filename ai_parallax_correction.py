#!/usr/bin/env python3
"""
AI-Powered Parallax Correction System
Dynamically adjusts crosshair positioning based on:
- Object distance estimation (using bounding box size)
- Camera-to-servo offset compensation
- Digital zoom compensation
- Object type-specific corrections
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional

class AIParallaxCorrector:
    def __init__(self, 
                 camera_height_mm: float = 50,  # Camera height above servo pivot
                 servo_offset_mm: float = 30,   # Horizontal offset between camera and servo
                 camera_fov_h: float = 60,      # Horizontal FOV in degrees
                 camera_fov_v: float = 45):     # Vertical FOV in degrees
        """
        Initialize AI parallax correction system
        
        Args:
            camera_height_mm: Vertical distance from camera to servo pivot point
            servo_offset_mm: Horizontal distance from camera to servo pivot point
            camera_fov_h: Camera horizontal field of view in degrees
            camera_fov_v: Camera vertical field of view in degrees
        """
        self.camera_height_mm = camera_height_mm
        self.servo_offset_mm = servo_offset_mm
        self.camera_fov_h = camera_fov_h
        self.camera_fov_v = camera_fov_v
        
        # Smoothing for stable crosshair
        self.last_crosshair_x = None
        self.last_crosshair_y = None
        self.smoothing_factor = 0.7  # Higher = more smoothing
        
        # Drone-only detection mode
        self.drone_only_mode = True
        self.min_confidence_drone = 0.3  # Minimum confidence for drones
        
        # Known object sizes for distance estimation (in mm)
        self.object_sizes = {
            'clock': 200,      # Wall clock ~20cm
            'bird': 150,       # Small bird ~15cm
            'kite': 500,       # Small kite ~50cm
            'frisbee': 270,    # Standard frisbee 27cm
            'sports ball': 220, # Soccer ball ~22cm
            'cell phone': 150, # Phone ~15cm
            'remote': 180,     # TV remote ~18cm
            'donut': 80,       # Donut ~8cm
            'apple': 70,       # Apple ~7cm
            'orange': 75,      # Orange ~7.5cm
            'drone': 300,      # Estimated drone size ~30cm
        }
        
        # Confidence weights for distance estimation
        self.confidence_weights = {
            'high': 1.0,    # >0.8 confidence
            'medium': 0.8,  # 0.4-0.8 confidence  
            'low': 0.6      # <0.4 confidence
        }
        
    def estimate_distance(self, bbox_width: int, bbox_height: int, 
                         object_class: str, confidence: float,
                         frame_width: int, frame_height: int) -> float:
        """
        Estimate distance to object using AI analysis
        
        Args:
            bbox_width: Bounding box width in pixels
            bbox_height: Bounding box height in pixels
            object_class: YOLO detected class name
            confidence: YOLO confidence score
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Estimated distance in millimeters
        """
        # Get expected real-world size
        expected_size_mm = self.object_sizes.get(object_class, 200)  # Default 20cm
        
        # Use larger dimension for more accurate distance estimation
        bbox_size_pixels = max(bbox_width, bbox_height)
        frame_size_pixels = max(frame_width, frame_height)
        
        # Calculate angular size
        angular_size_degrees = (bbox_size_pixels / frame_size_pixels) * max(self.camera_fov_h, self.camera_fov_v)
        angular_size_radians = math.radians(angular_size_degrees)
        
        # Distance estimation using similar triangles
        estimated_distance_mm = expected_size_mm / (2 * math.tan(angular_size_radians / 2))
        
        # Apply confidence weighting
        if confidence > 0.8:
            weight = self.confidence_weights['high']
        elif confidence > 0.4:
            weight = self.confidence_weights['medium']
        else:
            weight = self.confidence_weights['low']
            
        # Confidence-weighted distance with fallback
        base_distance = 1000  # 1 meter fallback
        weighted_distance = (estimated_distance_mm * weight) + (base_distance * (1 - weight))
        
        # Clamp to reasonable range (10cm to 10m)
        return max(100, min(10000, weighted_distance))
    
    def calculate_parallax_correction(self, distance_mm: float, 
                                    target_x_pixels: int, target_y_pixels: int,
                                    frame_width: int, frame_height: int,
                                    zoom_factor: float = 1.0) -> Tuple[int, int]:
        """
        Calculate AI-powered parallax correction
        
        Args:
            distance_mm: Estimated distance to target in mm
            target_x_pixels: Target X position in pixels
            target_y_pixels: Target Y position in pixels
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            zoom_factor: Current digital zoom factor
            
        Returns:
            Tuple of (corrected_x, corrected_y) in pixels
        """
        # Convert pixel positions to angles from center
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        # Account for zoom factor in angle calculations
        effective_fov_h = self.camera_fov_h / zoom_factor
        effective_fov_v = self.camera_fov_v / zoom_factor
        
        # Calculate angles from center
        angle_x_degrees = ((target_x_pixels - center_x) / frame_width) * effective_fov_h
        angle_y_degrees = ((target_y_pixels - center_y) / frame_height) * effective_fov_v
        
        # Convert to radians
        angle_x_rad = math.radians(angle_x_degrees)
        angle_y_rad = math.radians(angle_y_degrees)
        
        # Calculate 3D position of target relative to camera
        target_x_3d = distance_mm * math.tan(angle_x_rad)
        target_y_3d = distance_mm * math.tan(angle_y_rad)
        target_z_3d = distance_mm
        
        # Transform to servo coordinate system
        # Account for camera-servo offset
        servo_target_x = target_x_3d - self.servo_offset_mm
        servo_target_y = target_y_3d - self.camera_height_mm
        servo_target_z = target_z_3d
        
        # Calculate corrected angles for servo
        corrected_angle_x = math.atan2(servo_target_x, servo_target_z)
        corrected_angle_y = math.atan2(servo_target_y, servo_target_z)
        
        # Convert back to degrees
        corrected_angle_x_deg = math.degrees(corrected_angle_x)
        corrected_angle_y_deg = math.degrees(corrected_angle_y)
        
        # Convert corrected angles back to pixel coordinates for crosshair
        corrected_x_pixels = center_x + (corrected_angle_x_deg / effective_fov_h) * frame_width
        corrected_y_pixels = center_y + (corrected_angle_y_deg / effective_fov_v) * frame_height
        
        # Clamp to frame bounds
        corrected_x_pixels = max(0, min(frame_width - 1, corrected_x_pixels))
        corrected_y_pixels = max(0, min(frame_height - 1, corrected_y_pixels))
        
        return int(corrected_x_pixels), int(corrected_y_pixels)
    
    def get_adaptive_crosshair(self, detections: list, frame_width: int, frame_height: int,
                              zoom_factor: float = 1.0) -> Tuple[int, int, Dict]:
        """
        Generate AI-adaptive crosshair position based on detected objects
        
        Args:
            detections: List of YOLO detections with bbox, class, confidence
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            zoom_factor: Current digital zoom factor
            
        Returns:
            Tuple of (crosshair_x, crosshair_y, debug_info)
        """
        # Filter detections for STRICT drone-only mode
        if self.drone_only_mode:
            # Only these objects can be drones - NO furniture/appliances (expanded list)
            strict_drone_classes = {
                # Primary drone-like objects
                'clock', 'bird', 'kite', 'frisbee', 'sports ball', 'remote',
                # Additional objects that drones might be misclassified as
                'donut', 'apple', 'orange', 'cell phone', 'mouse', 'cup', 'bottle'
            }
            filtered_detections = []
            for detection in detections:
                class_name = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0)
                
                # STRICT filtering - only actual drone-like objects
                if (class_name in strict_drone_classes and 
                    confidence >= self.min_confidence_drone):
                    filtered_detections.append(detection)
            detections = filtered_detections
        
        if not detections:
            # No targets - return smoothed center
            center_x = int(frame_width / 2)
            center_y = int(frame_height / 2)
            
            # Apply smoothing if we had a previous position
            if self.last_crosshair_x is not None:
                center_x = int(self.last_crosshair_x * self.smoothing_factor + center_x * (1 - self.smoothing_factor))
                center_y = int(self.last_crosshair_y * self.smoothing_factor + center_y * (1 - self.smoothing_factor))
            
            self.last_crosshair_x = center_x
            self.last_crosshair_y = center_y
            return center_x, center_y, {"status": "no_targets", "zoom_factor": zoom_factor}
        
        # Find the most likely drone target
        best_detection = None
        best_score = 0
        
        for detection in detections:
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            
            # Priority scoring for drone-like objects
            priority_multiplier = 1.0
            if class_name in ['clock', 'bird', 'kite', 'frisbee']:
                priority_multiplier = 2.0
            elif class_name in ['sports ball', 'remote', 'cell phone']:
                priority_multiplier = 1.5
                
            score = confidence * priority_multiplier
            
            if score > best_score:
                best_score = score
                best_detection = detection
        
        if not best_detection:
            center_x = int(frame_width / 2)
            center_y = int(frame_height / 2)
            return center_x, center_y, {"status": "no_valid_targets"}
        
        # Extract detection data
        bbox = best_detection['bbox']  # [x1, y1, x2, y2]
        class_name = best_detection['class']
        confidence = best_detection['confidence']
        
        # Calculate target center
        target_x = int((bbox[0] + bbox[2]) / 2)
        target_y = int((bbox[1] + bbox[3]) / 2)
        
        # Calculate bounding box dimensions
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        # Estimate distance using AI
        estimated_distance = self.estimate_distance(
            bbox_width, bbox_height, class_name, confidence, frame_width, frame_height
        )
        
        # For close objects, use less correction to avoid over-correction
        if estimated_distance < 500:  # Less than 50cm
            # Use mostly the raw target position for very close objects
            corrected_x = int(target_x * 0.8 + (frame_width / 2) * 0.2)
            corrected_y = int(target_y * 0.8 + (frame_height / 2) * 0.2)
        else:
            # Calculate parallax correction for distant objects
            corrected_x, corrected_y = self.calculate_parallax_correction(
                estimated_distance, target_x, target_y, frame_width, frame_height, zoom_factor
            )
        
        # Apply smoothing to prevent blinking
        if self.last_crosshair_x is not None:
            smoothed_x = int(self.last_crosshair_x * self.smoothing_factor + corrected_x * (1 - self.smoothing_factor))
            smoothed_y = int(self.last_crosshair_y * self.smoothing_factor + corrected_y * (1 - self.smoothing_factor))
        else:
            smoothed_x = corrected_x
            smoothed_y = corrected_y
        
        self.last_crosshair_x = smoothed_x
        self.last_crosshair_y = smoothed_y
        
        # Debug information
        debug_info = {
            "status": "ai_corrected",
            "target_class": class_name,
            "confidence": confidence,
            "estimated_distance_mm": estimated_distance,
            "estimated_distance_m": estimated_distance / 1000,
            "original_position": (target_x, target_y),
            "corrected_position": (corrected_x, corrected_y),
            "smoothed_position": (smoothed_x, smoothed_y),
            "correction_offset": (smoothed_x - target_x, smoothed_y - target_y),
            "zoom_factor": zoom_factor,
            "bbox_size": (bbox_width, bbox_height)
        }
        
        return smoothed_x, smoothed_y, debug_info

# Global instance for use in main detection script
ai_parallax = None  # Will be initialized with config

def get_ai_crosshair_position(yolo_results, frame_width: int, frame_height: int, 
                             zoom_factor: float = 1.0) -> Tuple[int, int, Dict]:
    """
    Convenience function for integration with existing YOLO detection code
    
    Args:
        yolo_results: YOLO detection results
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        zoom_factor: Current digital zoom factor
        
    Returns:
        Tuple of (crosshair_x, crosshair_y, debug_info)
    """
    # Convert YOLO results to our format
    detections = []
    
    if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
        for i, box in enumerate(yolo_results.boxes):
            if hasattr(box, 'xyxy') and hasattr(box, 'conf') and hasattr(box, 'cls'):
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name from model
                if hasattr(yolo_results, 'names'):
                    class_name = yolo_results.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                detection = {
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'class': class_name
                }
                detections.append(detection)
    
    return ai_parallax.get_adaptive_crosshair(detections, frame_width, frame_height, zoom_factor)

if __name__ == "__main__":
    # Test the parallax correction system
    corrector = AIParallaxCorrector()
    
    # Test with simulated detection
    test_detections = [{
        'bbox': [500, 300, 600, 400],  # 100x100 pixel box
        'class': 'clock',
        'confidence': 0.85
    }]
    
    crosshair_x, crosshair_y, debug = corrector.get_adaptive_crosshair(
        test_detections, 1920, 1080, zoom_factor=2.0
    )
    
    print("AI Parallax Correction Test:")
    print(f"Corrected crosshair: ({crosshair_x}, {crosshair_y})")
    print(f"Debug info: {debug}")
