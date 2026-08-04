"""
recommendation.py
Rule-based shot recommendation engine.
Given opponent position and shot played, determines if a better option existed.
"""

OPPOSITE_ZONES = {
    "back-left":   "front-right",
    "back-right":  "front-left",
    "front-left":  "back-right",
    "front-right": "back-left"
}

# Future: expand this with more nuanced rules
# e.g. if opponent at net and you're at back -> clear or lob
# e.g. if opponent at back -> drop or net shot


def get_shot_zone(kps_flat, frame_w):
    """
    Approximates which side of the court the near player is hitting from
    using their hip keypoint x-position.
    Returns 'left' or 'right'.
    """
    hip_x = kps_flat[23 * 2] if len(kps_flat) > 47 else frame_w / 2
    return "left" if hip_x < frame_w / 2 else "right"


def evaluate(shot, opp_zone, shot_zone):
    """
    Evaluates whether the shot played was toward or away from the opponent.

    Returns:
        suggestion (str): feedback message
        is_good (bool): True if shot direction was good
        better_zone (str): where they should have played instead (if bad)
    """
    if opp_zone is None:
        return None, None, None

    # Extract which side opponent is on
    opp_side = "left" if "left" in opp_zone else "right"
    better_zone = OPPOSITE_ZONES[opp_zone]

    if opp_side == shot_zone:
        return f"Should play to {better_zone}", False, better_zone
    else:
        return "Good direction", True, None


def get_strategy_tip(shot, opp_zone):
    """
    Returns a more specific strategic tip based on shot type and opponent position.
    Placeholder for v0.2 expanded rule set.
    """
    tips = {
        ('clear', 'front-left'):  "Opponent at net — drop would be better than clear",
        ('clear', 'front-right'): "Opponent at net — drop would be better than clear",
        ('smash', 'back-left'):   "Opponent deep — net drop could catch them off guard",
        ('smash', 'back-right'):  "Opponent deep — net drop could catch them off guard",
        ('net_shot', 'front-left'):  "Opponent at net — drive or lift to push them back",
        ('net_shot', 'front-right'): "Opponent at net — drive or lift to push them back",
    }
    return tips.get((shot, opp_zone), None)
