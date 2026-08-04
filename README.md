# BadmintonAI — Gameplay Analysis & Shot Recommendation System

> *An AI-powered tool that analyzes badminton footage, classifies shots, tracks player positions, and recommends better shot choices in real time.*

---

## Vision

Most players improve by drilling the same shots repeatedly. But the bigger gap between amateur and pro isn't technique — it's **decision-making**. Pros know where to play the shuttle based on where their opponent is, where they just came from, and what the rally pattern looks like.

BadmintonAI watches your footage and tells you when you made a suboptimal shot choice and what you should have played instead. It's a coach that never gets tired and can review every single rally.

---

## Current State (v0.1)

### What works
- **Player detection** — YOLOv8 bounding boxes on both players, filtered to exclude umpires and crowd using aspect ratio + position heuristics
- **Pose estimation** — YOLOv8-pose skeleton overlay with 17 keypoints (shoulders, elbows, wrists, hips, knees, ankles)
- **Shuttlecock detection** — Roboflow-hosted YOLO model trained on badminton footage (~88% confidence on clean frames)
- **Shot classification** — RandomForest classifier trained on 104 manually labeled frames from pro match footage (Axelsen + Momota), 6 shot types, 76% overall accuracy
- **Contact frame detection** — Wrist velocity thresholding to trigger classification only during actual swings
- **Opponent position tracking** — Court zone system (front/back × left/right) derived from far-player bounding box center
- **Shot recommendation (v1)** — Rule-based: flags when near player hits toward opponent's zone, suggests opposite zone

### Shot types classified
`smash` `clear` `drop` `net_shot` `backhand` `backhand_serve`

### Known limitations
- Smash/drop confusion (~29% drop accuracy) — body position too similar at single contact frame
- Model is angle-specific — trained on low side-on camera only
- Pipeline not real-time — currently processes ~1 frame/second on Colab T4
- Amateur footage not yet tested
- Shuttlecock tracking unreliable on fast or motion-blurred frames

---

## Repo Structure

```
badminton-ai/
│
├── README.md
│
├── notebooks/
│   ├── 01_player_pose_detection.ipynb       # Player detection + skeleton overlay
│   ├── 02_shuttlecock_detection.ipynb       # Roboflow shuttle model integration
│   ├── 03_frame_scrubber.ipynb              # Interactive frame labeling tool
│   ├── 04_keypoint_extraction.ipynb         # Extract keypoints → CSV dataset
│   ├── 05_shot_classifier_training.ipynb    # Train + evaluate RandomForest
│   └── 06_full_pipeline.ipynb               # End-to-end analysis pipeline
│
├── src/
│   ├── player_detection.py                  # is_player() filter + box sorting
│   ├── shot_classifier.py                   # Classifier training + inference
│   ├── recommendation.py                    # Court zone logic + shot suggestions
│   └── pipeline.py                          # Main video processing loop
│
├── data/
│   └── labels/
│       ├── dataset_notes.txt                # Labeling decisions + known issues
│       └── keypoints.csv                    # 104-frame labeled dataset
│
└── models/
    └── shot_classifier.pkl                  # Trained RandomForest model
```

---

## Quickstart

### Requirements
```bash
pip install ultralytics roboflow opencv-python scikit-learn pandas numpy joblib
```

### Run the full pipeline on a video
```python
# In notebooks/06_full_pipeline.ipynb
# Upload your video to Colab and set the filename below

VIDEO_PATH = 'your_match.mp4'
OUTPUT_PATH = 'output_analysis.mp4'
START_SEC = 0       # skip intro/warmup
DURATION_SEC = 30   # how many seconds to process
```

### Train your own classifier
```python
# In notebooks/05_shot_classifier_training.ipynb
# Requires keypoints.csv in data/labels/
```

---

## Data

Training data was manually labeled from:
- **Victor Axelsen** — BWF match footage, low side-on fixed camera
- **Kento Momota** — highlight reel, same angle, used specifically to capture drop/clear examples

Labeling methodology:
- Single contact frame per shot (frame immediately before shuttle impact)
- Near-side player only (far player keypoints too noisy)
- Labels: `smash`, `clear`, `drop`, `net_shot`, `backhand`, `backhand_serve`

Known dataset biases documented in `data/labels/dataset_notes.txt`.

---

## Roadmap

### v0.2 (Summer 2025)
- [ ] Sequence-based features (5-frame windows) to fix smash/drop confusion
- [ ] Expand dataset to 300+ labeled frames across 3+ players
- [ ] Test on amateur footage and document failure modes
- [ ] TrackNet integration for reliable shuttlecock trajectory tracking
- [ ] Rule-based recommendation engine with 8-10 strategy rules

### v0.3 (Fall 2025 — UMich)
- [ ] LSTM/Transformer sequence classifier to replace RandomForest
- [ ] Rally-level context (shot history, player fatigue proxy, score state)
- [ ] Multi-angle robustness via joint-angle features instead of raw x/y keypoints
- [ ] Investigate NVIDIA LocateAnything-3B for shuttlecock + player detection

### v1.0 MVP (Spring 2026)
- [ ] ML-based recommendation engine trained on pro rally outcome data
- [ ] Streamlit web interface: upload video → get annotated output
- [ ] Real user testing with UMich badminton club players
- [ ] Performance optimization for faster-than-realtime processing

### Long-term vision
- Real-time analysis during live play
- Opponent tendency modeling across multiple matches
- Mobile app with on-device inference
- Full shot taxonomy (cross-court vs straight, jump smash, deceptive drops)

---

## Technical Notes

### Why RandomForest over deep learning?
With only 104 training samples, a deep network would overfit severely. RandomForest handles small tabular datasets well and gives interpretable feature importances — useful for understanding which joints matter most for each shot type.

### Why single-frame keypoints fail for smash/drop
A smash and drop look nearly identical at the contact frame — both involve a full overhead arm extension. The difference is wrist deceleration pattern across the swing arc, which requires sequential frame data to detect. This is the primary motivation for moving to sequence-based features in v0.2.

### Court zone system
The court is divided into 4 zones (front-left, front-right, back-left, back-right) derived from the far player's bounding box center position relative to frame dimensions. This is a rough approximation — a proper implementation would use court homography to map pixel coordinates to real-world court positions.

---

## Built With
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — player detection + pose estimation
- [Roboflow](https://roboflow.com) — shuttlecock detection model
- [scikit-learn](https://scikit-learn.org) — RandomForest classifier
- [OpenCV](https://opencv.org) — video processing
- [Google Colab](https://colab.research.google.com) — development environment (T4 GPU)

---

## Author
Built as a passion project exploring sports computer vision and ML-based tactical analysis. Development started summer 2025.
