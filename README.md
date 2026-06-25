# 🤟 SignBridge — Two-Way Real-Time Sign Language Interpreter

A full-stack, hardware-integrated system that bridges communication between signing and non-signing users in real time. SignBridge works in **two directions simultaneously**:

- **Sign → Text:** A webcam reads ASL hand gestures using computer vision and displays the detected letters as a live transcript.
- **Text → Sign:** A user types (or speaks) any word or phrase, and a **physical 3D-printed robotic hand** fingerspells it back — letter by letter — using servo motors.

> 🎥 **[Watch the Demo](#demo-video)**

---








## ✨ Features

**Sign → Text (Computer Vision)**
- Real-time ASL fingerspelling recognition via webcam at ~5 FPS
- Detects all 26 letters of the alphabet + the ILY (I Love You) handshape
- Transcript state machine with confirmation logic — requires a sign to be held for several frames before committing, eliminating jitter and false positives
- Automatic space insertion after a configurable no-hand gap
- Weighted nearest-neighbour classification with per-letter geometric disambiguation for similar signs (e.g. U vs V vs R, C vs O, K vs P)

**Text → Sign (Robotic Hand)**
- 3D-printed tendon-driven left hand controlled by 5 MG90S micro-servos via a PCA9685 PWM driver on Raspberry Pi 4
- Smooth servo interpolation between positions to prevent tendon strain and mechanical snapping
- Signs any typed or spoken phrase letter by letter with a neutral reset between each character
- Concurrent signing lock — new requests are queued gracefully while the robot is busy
- Graceful mock fallback — the full app (UI, vision, API) runs on non-RPi hardware for development and testing

**Web Interface**
- Split-panel UI: Sign-to-Text on the left, Text-to-Sign on the right
- Live finger state visualiser — shows each of the robot's 5 fingers as closed / half / extended in real time
- Letter-by-letter progress bar during signing
- Quick-phrase buttons for common greetings and responses
- Voice input (Web Speech API) — speak a phrase and the robot signs it
- Real-time Socket.IO connection with a live status badge

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Server | Python 3, Flask, Flask-SocketIO |
| Computer Vision | MediaPipe HandLandmarker (Tasks API), OpenCV, NumPy |
| Hardware Control | Adafruit CircuitPython PCA9685 & ServoKit (Raspberry Pi 4) |
| Frontend | HTML5, CSS3, Vanilla JavaScript (ES6+), Socket.IO client |
| Real-time Comms | WebSockets via Socket.IO |
| Voice Input | Web Speech API |

---

## 🦾 Hardware

The physical hand is a custom **tendon-driven** design:

- **3D-printed** based on the dimensions of a real left hand
- **Fingers** use a nylon thread tendon routed through the joints — a servo pulling the tendon curls the finger; relaxing it allows the finger to extend
- **5 × MG90S micro-servos** — one per finger (thumb, index, middle, ring, pinky)
- **PCA9685 16-channel PWM driver** communicates with the Raspberry Pi 4 over I2C
- Servo pulse width range configured to 500–2500 µs for full MG90S range of motion

**Servo → Finger Channel Mapping**

| Channel | Finger | Extended (relaxed) | Closed (pulled) |
|---|---|---|---|
| 0 | Thumb | 170° | 10° |
| 1 | Index | 170° | 10° |
| 2 | Middle | 170° | 10° |
| 3 | Ring | 170° | 10° |
| 4 | Pinky | 170° | 10° |

---

## 📁 Project Structure

```
signbridge/
├── app.py                  # Flask + Socket.IO server, signing thread management
├── vision.py               # MediaPipe landmark detection & ASL classification
├── robot.py                # PCA9685 servo controller with smooth interpolation
├── requirements.txt        # Python dependencies
├── download_model.py       # One-time script to download the MediaPipe model
│
├── templates/
│   └── index.html          # Main dual-panel web UI
│
└── static/
    ├── css/
    │   └── style.css       # Dark glassmorphism design
    └── js/
        └── main.js         # Socket.IO client, camera capture, state machine
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- `pip`
- A webcam
- *(For hardware signing)* Raspberry Pi 4 with PCA9685 board and 5 MG90S servos wired up

### 1. Clone the repository
```bash
git clone https://github.com/vijay-tharun-d/signbridge.git
cd signbridge
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the MediaPipe hand model
The hand landmark model is not included in the repo due to file size. Run this once to download it:
```bash
python download_model.py
```

### 5. Run the server
```bash
python app.py
```

Open your browser and go to **http://localhost:5000**

> **Running without a Raspberry Pi?** No problem — `robot.py` automatically detects if the hardware libraries are unavailable and falls into mock mode. Servo commands are logged to the console instead. The full UI, camera, and vision pipeline still work normally.

---

## 🎥 Demo Video

<div align="center">
  <video src="https://github.com/user-attachments/assets/0e5f3ac9-baee-4664-9a58-be7f01a0522e" controls width="720"></video>
</div>

> *The demo shows the robotic hand fingerspelling "HELLO" via the web interface, followed by the sign-to-text feature recognising letters from a webcam.*

---

## 🧠 How the Vision System Works

1. The browser captures webcam frames at ~5 FPS and sends each as a base64-encoded JPEG over WebSocket.
2. `vision.py` decodes the frame, runs MediaPipe's HandLandmarker to extract 21 3D landmarks per detected hand.
3. For each of the 5 fingers, a **continuous extension ratio** (0.0 = fully curled, 1.0 = fully extended) is computed using joint angle calculations (dot product at PIP and DIP joints), weighted 60%/40% respectively.
4. The ratios are compared against reference patterns for all 26 letters + ILY using **weighted Euclidean distance** in 5D feature space.
5. For sign pairs that are geometrically close (e.g. C vs O, B vs W, U vs V vs R), a secondary **geometric disambiguator** applies targeted ratio checks to resolve the ambiguity.
6. A **transcript state machine** on the frontend requires a sign to be held for 7 consecutive frames before appending it to the transcript, and enters a cooldown of 10 frames to prevent duplicates.

---

## 🧠 What I Learned

- Integrating hardware (Raspberry Pi, servo drivers, PWM) with a real-time software system over WebSockets
- Using **MediaPipe Tasks API** for hand landmark detection and computing continuous finger extension ratios from 3D joint geometry
- Building a thread-safe, real-time server with **Flask-SocketIO** using background threads and mutex locks for concurrent hardware access
- Designing a **tendon-driven robotic mechanism** and calibrating servo angles to map to reliable ASL finger positions
- Implementing a **finite state machine** on the frontend for noise-resilient gesture recognition (instead of naive frame-by-frame classification)

---

## ⚠️ Known Limitations

- **J and Z** involve wrist motion arcs that cannot be represented as static servo positions. J is approximated as a modified I; Z as a modified index-extended pose.
- Classification accuracy is highest under good lighting with a plain background.
- The MediaPipe model file (`hand_landmarker.task`) must be downloaded separately — see setup step 4.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
