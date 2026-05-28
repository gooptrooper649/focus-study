# 🧠 Focus AI – Distraction Detection System

An AI-powered productivity assistant that detects when a user gets distracted during study sessions using real-time computer vision and machine learning.

---

## 🚀 Overview

Focus AI is designed to improve concentration by monitoring gaze direction, detecting phone usage, and enforcing structured study sessions using a Pomodoro timer.

The system uses your webcam to analyze attention patterns and provides real-time feedback when distraction is detected.

---

## ✨ Features

* 👁️ **Real-Time Gaze Tracking**
  Detects where you are looking using facial landmarks

* 🎯 **Calibration System**
  Learns your natural study position for accurate tracking

* 📱 **Phone Detection (Strict Rule)**
  Uses object detection to identify mobile phone usage

* ⏱️ **Pomodoro Timer Integration**
  45-minute focus sessions + 15-minute breaks

* ⚠️ **Distraction Detection Engine**
  Tracks sustained gaze deviation

* 🧠 **Grace Period System**
  Allows short natural breaks before triggering alerts

* 🔊 **Alert System**
  Notifies user when distraction persists

* 🔒 **Privacy-Friendly**
  Runs completely locally (no data stored or uploaded)

---

## 🧠 How It Works

The system combines multiple components:

1. **Face & Eye Tracking**

   * Uses MediaPipe to detect facial landmarks
   * Calculates eye center positions

2. **Gaze Analysis**

   * Compares current gaze with calibrated reference point
   * Determines deviation threshold

3. **Object Detection**

   * Uses YOLOv8 to detect presence of a mobile phone

4. **Decision Engine**

   * Combines gaze + object detection + time logic
   * Applies rules for distraction detection

5. **Pomodoro Controller**

   * Manages work/break cycles automatically

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* YOLOv8 (Ultralytics)
* NumPy
* Pygame

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/focus-ai.git
cd focus-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download YOLO model

The model will automatically download on first run.

### 4. Add alert sound

Place a `beep.wav` file in the project root directory.

---

## ▶️ Usage

```bash
python main.py
```

### Controls:

* Press **C** → Calibrate focus position
* Press **ESC** → Exit

---

## ⚙️ Configuration

You can tweak these parameters in the code:

```python
DISTRACTION_THRESHOLD = 3   # seconds
GAZE_TOLERANCE = 60        # sensitivity
GRACE_LIMIT = 60           # seconds
WORK_TIME = 45 * 60
BREAK_TIME = 15 * 60
```

---

## ⚠️ Limitations

* Performance depends on hardware (YOLO can be heavy)
* Lighting conditions affect detection accuracy
* Single-user tracking only
* Basic UI (console + OpenCV window)

---

## 🔮 Future Improvements

* 📊 Focus analytics dashboard
* ⚡ Performance optimization (frame skipping / async YOLO)
* 🧠 Adaptive learning system
* 🖥️ GUI interface
* ☁️ Optional cloud sync

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 💡 Inspiration

Built to solve a personal problem: staying focused in a world full of distractions.
