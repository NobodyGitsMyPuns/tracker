@echo off
echo 🚁 ESP32 Drone Tracker - OTA Upload Script
echo ==========================================

echo.
echo 📡 Step 1: Connect ESP32 via USB and upload initial code
echo.
pause

echo 🔍 Compiling ESP32 code...
arduino-cli compile --fqbn esp32:esp32:esp32 ESP32_DroneTracker_OTA

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Compilation failed!
    pause
    exit /b 1
)

echo ✅ Compilation successful!
echo.

echo 📤 Uploading to ESP32 via USB...
arduino-cli upload -p COM3 --fqbn esp32:esp32:esp32 ESP32_DroneTracker_OTA

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Upload failed! Check COM port and connection.
    echo 💡 Try different COM ports: COM4, COM5, etc.
    pause
    exit /b 1
)

echo ✅ Upload successful!
echo.
echo 📊 Opening Serial Monitor to get IP address...
echo    Press Ctrl+C to stop monitoring when you see the IP address
echo.
pause

arduino-cli monitor -p COM3 -c baudrate=115200

echo.
echo 🎯 Next steps:
echo    1. Note the IP address from serial monitor
echo    2. Open browser to http://YOUR_ESP32_IP/update
echo    3. Upload new firmware wirelessly via OTA
echo.
pause
