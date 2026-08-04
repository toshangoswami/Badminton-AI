"""
pipeline.py
Main video processing loop.
Combines player detection, pose estimation, shot classification,
and recommendation into a single annotated output video.

Usage (in Colab):
    from src.pipeline import run
    run('my_match.mp4', 'output.mp4', start_sec=10, duration_sec=30)
"""

import cv2
import numpy as np
from ultralytics import YOLO

from player_detection import is_player, sort_players, get_opponent_zone
from shot_classifier import load as load_classifier, wrist_velocity, predict
from recommendation import evaluate

# ── Constants ──────────────────────────────────────────────────────────────
VELOCITY_THRESHOLD = 15    # minimum wrist pixel displacement to trigger classification
DISPLAY_FRAMES = 45        # how long to show shot label after detection (~1.5s at 30fps)


def run(
    video_path,
    output_path='output_analysis.mp4',
    start_sec=0,
    duration_sec=30,
    model_path='models/shot_classifier.pkl',
    velocity_threshold=VELOCITY_THRESHOLD
):
    """
    Processes a badminton video and outputs an annotated version.

    Args:
        video_path (str): path to input video
        output_path (str): path for annotated output video
        start_sec (int): seconds to skip at the start (warmup, intros)
        duration_sec (int): how many seconds to process
        model_path (str): path to trained shot classifier
        velocity_threshold (float): wrist speed threshold for shot detection
    """

    # Load models
    pose_model = YOLO('yolov8n-pose.pt')
    detect_model = YOLO('yolov8n.pt')
    clf = load_classifier(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
    max_frames = int(fps * duration_sec)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # State
    frame_count = 0
    prev_wrist_pos = None
    opp_zone = None
    last_shot = None
    last_suggestion = None
    last_color = (0, 255, 0)
    shot_display_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        annotated = frame.copy()

        # ── Player detection ────────────────────────────────────────────────
        detect_results = detect_model(frame, classes=[0], verbose=False)
        all_boxes = detect_results[0].boxes.xyxy.cpu().numpy()
        boxes = [b for b in all_boxes if is_player(b, frame_w, frame_h)]

        near_box, far_box = sort_players(boxes, frame_w, frame_h)

        if far_box is not None:
            opp_zone = get_opponent_zone(far_box, frame_w, frame_h)

            # Draw far player (opponent) in red
            cv2.rectangle(annotated,
                         (int(far_box[0]), int(far_box[1])),
                         (int(far_box[2]), int(far_box[3])),
                         (0, 0, 255), 2)
            opp_cx = int((far_box[0] + far_box[2]) / 2)
            cv2.putText(annotated, f"Opponent: {opp_zone}",
                       (opp_cx - 60, int(far_box[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if near_box is not None:
            # Draw near player (hitter) in green
            cv2.rectangle(annotated,
                         (int(near_box[0]), int(near_box[1])),
                         (int(near_box[2]), int(near_box[3])),
                         (0, 255, 0), 2)

        # ── Pose + shot classification ───────────────────────────────────────
        pose_results = pose_model(frame, verbose=False)
        if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.xy) > 0:
            kps = pose_results[0].keypoints.xy[0].cpu().numpy()
            kps_flat = kps.flatten()

            if kps_flat.max() > 0:
                velocity = wrist_velocity(kps_flat, prev_wrist_pos)
                prev_wrist_pos = (kps_flat[16 * 2], kps_flat[16 * 2 + 1])

                if velocity > velocity_threshold:
                    shot = predict(clf, kps_flat)
                    shot_zone = "left" if kps_flat[23 * 2] < frame_w / 2 else "right"

                    if opp_zone is not None and len(boxes) >= 2:
                        suggestion, is_good, _ = evaluate(shot, opp_zone, shot_zone)
                        color = (0, 255, 0) if is_good else (0, 0, 255)

                        last_shot = shot
                        last_suggestion = suggestion
                        last_color = color
                        shot_display_frames = DISPLAY_FRAMES

        # ── Overlay text ─────────────────────────────────────────────────────
        if shot_display_frames > 0:
            if last_shot:
                cv2.putText(annotated, f"Shot: {last_shot}",
                           (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, (255, 255, 0), 3)
            if last_suggestion:
                cv2.putText(annotated, last_suggestion,
                           (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                           1.0, last_color, 3)
            shot_display_frames -= 1

        out.write(annotated)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{max_frames} frames...")

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")
