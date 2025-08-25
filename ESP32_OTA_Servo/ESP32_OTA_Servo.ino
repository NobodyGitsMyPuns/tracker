#include <WiFi.h>
#include <ESPmDNS.h>
#include <NetworkUdp.h>
#include <ArduinoOTA.h>
#include <WebServer.h>
#include <Arduino_JSON.h>
#include "config.h"  // Auto-generated from Python config

// Use config values
const char *ssid = WIFI_SSID;
const char *password = WIFI_PASSWORD;

// Servo configuration now comes from config.h (auto-generated)
// PWM, timing, and pin definitions are in config.h

// Home positions from config (can still be changed via web API)
int pan_home = PAN_HOME;     // From config.h
int tilt_home = TILT_HOME;   // From config.h

// Movement variables
int panAngle = CENTER_ANGLE;
int tiltAngle = CENTER_ANGLE;

// Web server for drone tracking commands
WebServer server(80);

// Tracking mode
bool trackingMode = false;

void setup() {
  Serial.begin(115200);
  Serial.println("🚁 ESP32 OTA + Servo Sweep Starting...");
  
  // Setup PWM for servos (ESP32 Core 3.x)
  ledcAttach(PAN_PIN, PWM_FREQ, PWM_RES);
  ledcAttach(TILT_PIN, PWM_FREQ, PWM_RES);
  
  // Setup blaster relay pin
  pinMode(BLASTER_PIN, OUTPUT);
  digitalWrite(BLASTER_PIN, LOW);  // Relay off initially
  
  // Center servos to configured home positions
  writeServo(PAN_PIN, pan_home);
  writeServo(TILT_PIN, tilt_home);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }

  ArduinoOTA
    .onStart([]() {
      String type;
      if (ArduinoOTA.getCommand() == U_FLASH) {
        type = "sketch";
      } else {
        type = "filesystem";
      }
      Serial.println("Start updating " + type);
    })
    .onEnd([]() {
      Serial.println("\nEnd");
    })
    .onProgress([](unsigned int progress, unsigned int total) {
      Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
    })
    .onError([](ota_error_t error) {
      Serial.printf("Error[%u]: ", error);
      if (error == OTA_AUTH_ERROR) {
        Serial.println("Auth Failed");
      } else if (error == OTA_BEGIN_ERROR) {
        Serial.println("Begin Failed");
      } else if (error == OTA_CONNECT_ERROR) {
        Serial.println("Connect Failed");
      } else if (error == OTA_RECEIVE_ERROR) {
        Serial.println("Receive Failed");
      } else if (error == OTA_END_ERROR) {
        Serial.println("End Failed");
      }
    });

  ArduinoOTA.begin();

  // Setup web server routes for drone tracking
  setupWebServer();
  server.begin();

  Serial.println("✅ Ready with OTA + Web Server!");
  Serial.print("📍 IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println("🎯 Servos ready - NO auto-sweep, manual control only");
  Serial.println("📡 Web server endpoints:");
  Serial.println("   POST /track - Track drone: {\"x_offset\": px, \"y_offset\": px}");
  Serial.println("   GET  /center - Center servos");
  Serial.println("   GET  /sweep - Start sweep mode");
  Serial.println("   GET  /stop - Stop all movement");
  Serial.println("   GET  /status - Get current position");
}

void loop() {
  ArduinoOTA.handle();
  server.handleClient();  // Handle web requests
  
  // Debug: Print mode status every 5 seconds
  static unsigned long lastDebug = 0;
  if (millis() - lastDebug > 5000) {
    Serial.println("🔧 DEBUG: trackingMode=" + String(trackingMode));
    Serial.println("🔧 Current position: Pan=" + String(panAngle) + "°, Tilt=" + String(tiltAngle) + "°");
    lastDebug = millis();
  }
  
  // COMPLETELY DISABLE SWEEP - Only handle web requests
  if (false) {  // Force disable sweep regardless of sweepMode
    // Sweep code disabled
  } else {
    // Just handle web requests and stay still
    delay(10);
  }
}

void writeServo(int pin, int angle) {
  int pulseWidth = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  int dutyCycle = (pulseWidth * 65535) / (1000000 / PWM_FREQ);
  ledcWrite(pin, dutyCycle);
}

void setupWebServer() {
  // CORS headers for cross-origin requests
  server.onNotFound([]() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    server.send(404, "text/plain", "Not Found");
  });

  // Handle preflight OPTIONS requests
  server.on("/track", HTTP_OPTIONS, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "POST");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    server.send(200);
  });

  // 🎯 MAIN DRONE TRACKING ENDPOINT - Receives pixel offsets from YOLO
  server.on("/track", HTTP_POST, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    
    if (server.hasArg("plain")) {
      String body = server.arg("plain");
      
      // Parse JSON using Arduino_JSON
      JSONVar doc = JSON.parse(body);
      
      if (JSON.typeof(doc) == "undefined") {
        server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
        return;
      }
      
      // Get pixel offsets from camera center
      int x_offset = doc.hasOwnProperty("x_offset") ? (int)doc["x_offset"] : 0;
      int y_offset = doc.hasOwnProperty("y_offset") ? (int)doc["y_offset"] : 0;
      
      // Convert pixel offsets to servo angles (VERY fine tracking adjustments)
      float pan_adjustment = (x_offset / 100.0) * 1.0;     // Ultra fine: 100px = 1°
      float tilt_adjustment = (y_offset / 100.0) * 1.0;    // Ultra fine: 100px = 1°
      
      // Calculate new servo positions based on CURRENT position (not center!)
      int newPanAngle = constrain(panAngle + pan_adjustment, 0, 180);     // Move from current position
      int newTiltAngle = constrain(tiltAngle - tilt_adjustment, 20, 160); // Move from current position
      
      // Move servos to track drone
      panAngle = newPanAngle;
      tiltAngle = newTiltAngle;
      writeServo(PAN_PIN, panAngle);
      writeServo(TILT_PIN, tiltAngle);
      
      // Switch to tracking mode
      
      trackingMode = true;
      
      // Response
      String response = "{\"status\":\"tracking\",\"x_offset\":" + String(x_offset) + 
                       ",\"y_offset\":" + String(y_offset) + 
                       ",\"pan\":" + String(panAngle) + 
                       ",\"tilt\":" + String(tiltAngle) + "}";
      
      server.send(200, "application/json", response);
      
      Serial.println("🚁 TRACKING COMMAND RECEIVED!");
      Serial.println("   Input: X=" + String(x_offset) + "px, Y=" + String(y_offset) + "px");
      Serial.println("   Adjustments: Pan=" + String(pan_adjustment) + "°, Tilt=" + String(tilt_adjustment) + "°");
      Serial.println("   New Angles: Pan=" + String(newPanAngle) + "°, Tilt=" + String(newTiltAngle) + "°");
      Serial.println("   Servos moving NOW!");
    } else {
      server.send(400, "application/json", "{\"error\":\"No data received\"}");
    }
  });

  // Center servos to home positions
  server.on("/center", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    
    panAngle = pan_home;     // Use configured home position
    tiltAngle = tilt_home;   // Use configured home position
    writeServo(PAN_PIN, panAngle);
    writeServo(TILT_PIN, tiltAngle);
    
    trackingMode = false;
    
    server.send(200, "application/json", "{\"status\":\"centered\",\"pan\":" + String(panAngle) + ",\"tilt\":" + String(tiltAngle) + "}");
    Serial.println("🎯 Servos centered to home positions via web command");
  });



  // Stop all movement
      server.on("/stop", HTTP_GET, []() {
      server.sendHeader("Access-Control-Allow-Origin", "*");
      
      trackingMode = false;
    
    server.send(200, "application/json", "{\"status\":\"stopped\"}");
    Serial.println("⏸️ All movement stopped via web command");
  });

  // Fire blaster (300ms solid pulse)
  server.on("/fire", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    
    digitalWrite(BLASTER_PIN, HIGH);  // Fire!
    Serial.println("🔥 BLASTER ON - 300ms PULSE!");
    delay(300);  // Hold solid for 300ms
    digitalWrite(BLASTER_PIN, LOW);   // Stop firing
    Serial.println("⏹️ BLASTER OFF - PULSE COMPLETE!");
    
    server.send(200, "application/json", "{\"status\":\"fired\",\"duration\":300}");
  });

  // Get current status
  server.on("/status", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    
    String mode = (trackingMode ? "tracking" : "stopped");
    String response = "{\"pan\":" + String(panAngle) + 
                     ",\"tilt\":" + String(tiltAngle) + 
                     ",\"mode\":\"" + mode + "\"}";
    
    server.send(200, "application/json", response);
  });

  // Movement endpoints with step size parameter
  server.on("/left", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    int step = server.hasArg("step") ? server.arg("step").toInt() : 5;
    step = constrain(step, 1, 30);  // Safety limit
    panAngle -= step;
    panAngle = constrain(panAngle, 0, 180);
    writeServo(PAN_PIN, panAngle);
    Serial.println("🔧 LEFT " + String(step) + "°: Pan = " + String(panAngle) + "°");
    server.send(200, "text/plain", "LEFT " + String(step) + "°: Pan = " + String(panAngle) + "°");
  });
  
  server.on("/right", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    int step = server.hasArg("step") ? server.arg("step").toInt() : 5;
    step = constrain(step, 1, 30);
    panAngle += step;
    panAngle = constrain(panAngle, 0, 180);
    writeServo(PAN_PIN, panAngle);
    Serial.println("🔧 RIGHT " + String(step) + "°: Pan = " + String(panAngle) + "°");
    server.send(200, "text/plain", "RIGHT " + String(step) + "°: Pan = " + String(panAngle) + "°");
  });
  
  server.on("/up", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    int step = server.hasArg("step") ? server.arg("step").toInt() : 5;
    step = constrain(step, 1, 30);
    tiltAngle -= step;  // FIXED: UP = decrease angle (camera points up)
    tiltAngle = constrain(tiltAngle, 20, 160);
    writeServo(TILT_PIN, tiltAngle);
    Serial.println("🔧 UP " + String(step) + "°: Tilt = " + String(tiltAngle) + "°");
    server.send(200, "text/plain", "UP " + String(step) + "°: Tilt = " + String(tiltAngle) + "°");
  });
  
  server.on("/down", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    int step = server.hasArg("step") ? server.arg("step").toInt() : 5;
    step = constrain(step, 1, 30);
    tiltAngle += step;  // FIXED: DOWN = increase angle (camera points down)
    tiltAngle = constrain(tiltAngle, 20, 160);
    writeServo(TILT_PIN, tiltAngle);
    Serial.println("🔧 DOWN " + String(step) + "°: Tilt = " + String(tiltAngle) + "°");
    server.send(200, "text/plain", "DOWN " + String(step) + "°: Tilt = " + String(tiltAngle) + "°");
  });

  // Simple web interface
  server.on("/", HTTP_GET, []() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    
    String mode = (trackingMode ? "Tracking" : "Stopped");
    
    String html = "<!DOCTYPE html><html><head><title>ESP32 Drone Tracker</title></head><body>";
    html += "<h1>🚁 ESP32 Drone Tracker</h1>";
    html += "<p><strong>Current Position:</strong> Pan=" + String(panAngle) + "°, Tilt=" + String(tiltAngle) + "°</p>";
    html += "<p><strong>Mode:</strong> " + mode + "</p>";
    html += "<p><a href='/center'>Center Servos</a> | <a href='/stop'>Stop</a> | <a href='/fire'>🔥 FIRE!</a></p>";
    html += "<h3>API Endpoints:</h3>";
    html += "<ul>";
    html += "<li>POST /track - {\"x_offset\": px, \"y_offset\": px}</li>";
    html += "<li>GET /center - Center servos</li>";

    html += "<li>GET /stop - Stop movement</li>";
    html += "<li>GET /fire - Fire blaster (150ms)</li>";
    html += "<li>GET /status - Get current position</li>";
    html += "</ul>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
  });
}


