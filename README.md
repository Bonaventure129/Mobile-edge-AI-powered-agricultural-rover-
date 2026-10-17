# 🌿 Crop Scanner: Autonomous Edge-AI Disease Detector

**An offline, mobile robotics platform for real-time agricultural diagnostics.**

## 📖 Executive Summary

The **Crop Scanner** is an autonomous, teleoperated agricultural rover engineered to execute real-time, edge-AI computer vision diagnostics on crop foliage. Designed specifically to address the economic impact of Cassava plant diseases, the system utilizes a distributed "hardware-in-the-loop" architecture.

By operating a lightweight, quantized neural network directly on edge hardware, the rover functions **completely off-grid**. It broadcasts its own local Wi-Fi Access Point (AP), allowing operators to drive the rover and scan crops using any smartphone or tablet in remote farm environments where cellular data and internet access are unavailable.

## ✨ Core Capabilities

* **Offline Edge Inference:** Executes a locally hosted Google CropNet TensorFlow Lite model, optimized via INT8 post-training quantization, achieving <200ms inference times on a CPU.

* **Precision Teleoperation:** Ultra-low latency, bi-directional WebSocket communication (Port 81) enables responsive differential-drive motor control with a built-in 300ms dead-man's switch for safety.

* **Dual-Interface Output:**

  * *Remote:* A responsive, dark-mode Flask web dashboard serving an MJPEG video stream and unified control UI.

  * *Local:* A physical, chassis-mounted LCD screen running natively via Wayland/Wayfire for on-the-ground visual feedback.

* **Zero-Configuration Deployment:** Automated Linux desktop hooks bypass standard boot sequences, launching the camera driver, GUI, and web server autonomously upon receiving power.

## ⚙️ System Architecture

The rover utilizes a split-brain processing methodology to separate cognitive/visual tasks from hardware kinematics.

### Hardware Stack

| Component | Function / Specification | 
| ----- | ----- | 
| **Primary Compute (Brain)** | Raspberry Pi 4 Model B (Runs AI, Video Encoding, and Web Server) | 
| **Kinematic Node (Spine)** | ESP32-S3 Microcontroller (Handles PWM generation, WebSockets, AP Routing) | 
| **Motor Actuation** | L298N Dual H-Bridge Motor Driver | 
| **Vision System** | Raspberry Pi Camera Module V1 (OV5647), Native 640x480 hardware capture | 
| **Power Distribution** | 18650 Li-ion Cell configuration, routed through an LM2596 Buck Converter (Tuned to 5.1V) | 

### Software & ML Stack

* **Operating System:** Debian Linux (Trixie) with Wayfire/Wayland Compositor.

* **Backend Server:** Python 3.13, Flask (HTTP Port 5000), OpenCV (cv2).

* **Inference Engine:** `ai-edge-litert` (TensorFlow Lite XNNPACK CPU Delegate).

* **Model Architecture:** MobileNet-based CNN, mapping 224x224 RGB inputs to 6 output classes:

  1. Cassava Bacterial Blight

  2. Cassava Brown Streak

  3. Cassava Green Mottle

  4. Cassava Mosaic

  5. Healthy

  6. Unknown (Reject class)

## 🚀 Operating Instructions

### Phase 1: Boot Sequence

1. Locate the main power switch on the chassis and flip it to the **ON** position.

2. The system requires approximately **8 to 15 seconds** to supply voltage, boot the Debian kernel, and initialize the graphical desktop.

3. Once the physical LCD screen displays a full-screen live camera feed with a green targeting box, the system is fully operational.

### Phase 2: Network Handshake

1. On your control device (smartphone, tablet, or laptop), open the Wi-Fi settings.

2. Connect to the rover's secure local network:

   * **SSID:** `ROBOT-C3`

   * *Note: Your OS may issue a "No Internet Connection" warning. Dismiss this warning and remain connected to the network.*

3. Open a modern web browser and navigate to the embedded control server:

   * **URL:** `http://192.168.4.2:5000`

4. Confirm the dashboard loads and the telemetry badge in the upper right reads **ROBOT LINKED**.

### Phase 3: Field Navigation

* **Engage Drive:** Tap the **DRIVE OFF** toggle so it illuminates green and reads **DRIVE ON**. The directional pad will unlock.

* **Maneuvering:** Press and hold the directional arrows. The ESP32 utilizes skid-steer (differential) logic to navigate.

* **Speed Modulation:** Use the **+** and **-** buttons to dynamically scale the PWM output (0-100%).

* **Emergency Halt:** Tap the large red **STOP** button to sever motor power instantly.

### Phase 4: Diagnostic Scanning

For the neural network to output highly confident predictions, the visual data must mirror the model's training distribution.

1. Navigate the rover until the camera is facing the target plant.

2. Observe the green 80% boundary box labeled **"ALIGN LEAF IN BOX"**.

3. **Crucial Alignment:** Ensure *one single leaf* completely fills this bounding box.

4. **Crucial Lighting:** Ensure the leaf is front-lit (sunlight hitting the surface facing the camera). Avoid silhouetting the leaf against a bright sky.

5. Tap the **SCAN PLANT** button.

6. The dashboard will display the identified disease (or "Healthy") and its statistical confidence score. The result will simultaneously overlay on the rover's physical LCD screen for exactly 5 seconds.

## 🛠️ Diagnostics & Troubleshooting

| Symptom | Diagnosis | Resolution | 
| ----- | ----- | ----- | 
| **Dashboard stuck on "OFFLINE"** | Device disconnected from AP. | Smartphones often auto-switch to cellular data when Wi-Fi lacks internet. Force your phone to remain connected to `ROBOT-C3`. | 
| **Model returns high variance or "Unknown"** | Visual noise or poor framing. | Place a plain, solid-colored object (like a clipboard or paper) behind the leaf to isolate it from background clutter. Ensure the leaf is well-lit. | 
| **Physical screen remains blank** | Desktop initialization failure. | Wait 15 seconds. If still blank, verify the LM2596 buck converter is supplying exactly 5.1V to the Pi to prevent low-voltage throttling. | 
| **Motors do not respond to commands** | WebSocket timeout / Dead-man switch active. | Refresh the web page to re-establish the Port 81 WebSocket handshake with the ESP32. | 

### Acknowledgments & Credits

*Developed, engineered, and programmed by **Ugbah A. Bonaventure** — Conrad Robotics / Mechatronics Engineering Department, Federal University of Technology Owerri (FUTO).*