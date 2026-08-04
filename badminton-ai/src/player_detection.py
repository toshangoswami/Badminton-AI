"""
player_detection.py
Filters YOLO person detections to actual court players,
sorts by proximity, and returns near/far player boxes.
"""

import numpy as np


def is_player(box, frame_w, frame_h):
    """
    Returns True if a bounding box is likely a court player
    rather than an umpire, spectator, or background person.

    Filters by:
    - Aspect ratio (players are tall and narrow)
    - Vertical position (players are in the middle band of the frame)
    - Horizontal position (players are not at the far edges)
    - Minimum size (players take up at least 15% of frame height)
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Must be taller than wide (player silhouette)
    if height < width * 1.2:
        return False

    # Must be in the middle vertical band (not crowd at top, not bottom edge)
    if center_y < frame_h * 0.2 or center_y > frame_h * 0.95:
        return False

    # Must be in the middle horizontal band (not umpire at sides)
    if center_x < frame_w * 0.1 or center_x > frame_w * 0.9:
        return False

    # Must be a reasonable size relative to frame
    if height < frame_h * 0.15:
        return False

    return True


def sort_players(boxes, frame_w, frame_h):
    """
    Given a list of filtered player boxes, sorts by bounding box area
    (larger = closer to camera) and returns (near_box, far_box).
    Returns (None, None) if fewer than 2 players detected.
    """
    if len(boxes) < 2:
        return None, None

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    sorted_boxes = [b for _, b in sorted(zip(areas, boxes), reverse=True)]

    return sorted_boxes[0], sorted_boxes[1]


def get_opponent_zone(far_box, frame_w, frame_h):
    """
    Returns the court zone string for the far (opponent) player.
    Zones: 'front-left', 'front-right', 'back-left', 'back-right'
    """
    if far_box is None:
        return None

    cx = (far_box[0] + far_box[2]) / 2
    cy = (far_box[1] + far_box[3]) / 2

    left = cx < frame_w / 2
    back = cy < frame_h / 2

    if left and back:
        return "back-left"
    elif not left and back:
        return "back-right"
    elif left and not back:
        return "front-left"
    else:
        return "front-right"
