"""
shot_classifier.py
Trains and runs the RandomForest shot classifier.
Input: pose keypoints (17 joints x 2 coordinates = 34 features)
Output: shot type label
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

SHOT_LABELS = ['smash', 'clear', 'drop', 'net_shot', 'backhand', 'backhand_serve']

HEADER = [f'kp{i}_{c}' for i in range(17) for c in ['x', 'y']]


def train(csv_path='data/labels/keypoints.csv', save_path='models/shot_classifier.pkl'):
    """
    Trains a RandomForest classifier on labeled keypoint data.
    Prints classification report and saves model to disk.
    """
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1)
    y = df['label']

    print("Label distribution:")
    print(y.value_counts())
    print(f"\nTotal samples: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Results ---")
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

    return model


def load(model_path='models/shot_classifier.pkl'):
    """Loads a trained classifier from disk."""
    return joblib.load(model_path)


def wrist_velocity(kps_flat, prev_wrist_pos):
    """
    Computes pixel displacement of right wrist between frames.
    Used to detect when a shot is being played.
    """
    right_wrist_x = kps_flat[16 * 2]
    right_wrist_y = kps_flat[16 * 2 + 1]

    if prev_wrist_pos is None:
        return 0

    dx = right_wrist_x - prev_wrist_pos[0]
    dy = right_wrist_y - prev_wrist_pos[1]
    return np.sqrt(dx ** 2 + dy ** 2)


def predict(clf, kps_flat):
    """
    Predicts shot type from a flat array of 34 keypoint values.
    Returns shot label string.
    """
    df = pd.DataFrame([kps_flat], columns=HEADER)
    return clf.predict(df)[0]
