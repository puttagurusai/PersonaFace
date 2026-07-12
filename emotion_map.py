"""
emotion_map.py

Contains EMOTION_MAP: upper-face only blendshape targets for different emotions.
Jaw and full mouth interior shapes are deliberately excluded (they are controlled by lip-sync visemes).

Only brows, eyes, eye area, cheeks, and mouth corners are used here.
"""

NEUTRAL_REST = {
    # Jaw — natural slight opening, not clenched
    "jawOpen":              0.03,
    "jawForward":           0.0,
    "jawLeft":              0.0,
    "jawRight":             0.0,

    # Mouth — natural resting lip tone (very subtle)
    "mouthClose":           0.05,
    "mouthPucker":          0.0,
    "mouthFunnel":          0.0,
    "mouthRollLower":       0.02,
    "mouthRollUpper":       0.02,
    "mouthShrugLower":      0.0,
    "mouthShrugUpper":      0.0,
    "mouthSmileLeft":       0.0,
    "mouthSmileRight":      0.0,
    "mouthFrownLeft":       0.0,
    "mouthFrownRight":      0.0,
    "mouthDimpleLeft":      0.0,
    "mouthDimpleRight":     0.0,
    "mouthUpperUpLeft":     0.0,
    "mouthUpperUpRight":    0.0,
    "mouthLowerDownLeft":   0.0,
    "mouthLowerDownRight":  0.0,
    "mouthLeft":            0.0,
    "mouthRight":           0.0,
    "mouthStretchLeft":     0.0,
    "mouthStretchRight":    0.0,
    "mouthPressLeft":       0.0,
    "mouthPressRight":      0.0,

    # Brows — completely neutral
    "browDownLeft":         0.0,
    "browDownRight":        0.0,
    "browInnerUp":          0.0,
    "browOuterUpLeft":      0.0,
    "browOuterUpRight":     0.0,

    # Eyes — open but with tiny natural squint (not wide, not fully relaxed)
    "eyeBlinkLeft":         0.0,
    "eyeBlinkRight":        0.0,
    "eyeSquintLeft":        0.05,
    "eyeSquintRight":       0.05,
    "eyeWideLeft":          0.0,
    "eyeWideRight":         0.0,

    # Cheeks
    "cheekPuff":            0.0,
    "cheekSquintLeft":      0.0,
    "cheekSquintRight":     0.0,

    # Nose
    "noseSneerLeft":        0.0,
    "noseSneerRight":       0.0,

    # Tongue
    "tongueOut":            0.0,
}

EMOTION_MAP = {
    "neutral": NEUTRAL_REST.copy(),   # Use the natural rest pose as baseline

    "happy": {
        "browInnerUp": 0.1,
        "browOuterUpLeft": 0.15,
        "browOuterUpRight": 0.15,
        "eyeSquintLeft": 0.6,
        "eyeSquintRight": 0.6,
        "eyeBlinkLeft": 0.1,
        "eyeBlinkRight": 0.1,
        "cheekSquintLeft": 0.7,
        "cheekSquintRight": 0.7,
        "mouthSmileLeft": 0.85,
        "mouthSmileRight": 0.85,
    },

    "sad": {
        "browDownLeft": 0.55,
        "browDownRight": 0.55,
        "browInnerUp": 0.4,
        "eyeSquintLeft": 0.25,
        "eyeSquintRight": 0.25,
        "eyeBlinkLeft": 0.15,
        "eyeBlinkRight": 0.15,
        "mouthFrownLeft": 0.75,
        "mouthFrownRight": 0.75,
    },

    "angry": {
        "browDownLeft": 0.9,
        "browDownRight": 0.9,
        "browInnerUp": 0.1,
        "eyeSquintLeft": 0.65,
        "eyeSquintRight": 0.65,
        "mouthFrownLeft": 0.3,
        "mouthFrownRight": 0.3,
    },

    "surprised": {
        "browInnerUp": 0.95,
        "browOuterUpLeft": 0.6,
        "browOuterUpRight": 0.6,
        "eyeWideLeft": 0.85,
        "eyeWideRight": 0.85,
        "eyeBlinkLeft": 0.05,
        "eyeBlinkRight": 0.05,
        "mouthSmileLeft": 0.1,
        "mouthSmileRight": 0.1,
    },

    "disgusted": {
        "browDownLeft": 0.4,
        "browDownRight": 0.4,
        "eyeSquintLeft": 0.55,
        "eyeSquintRight": 0.55,
        "cheekSquintLeft": 0.4,
        "cheekSquintRight": 0.4,
        "mouthFrownLeft": 0.5,
        "mouthFrownRight": 0.5,
    },

    "fearful": {
        "browInnerUp": 0.7,
        "browOuterUpLeft": 0.3,
        "browOuterUpRight": 0.3,
        "eyeWideLeft": 0.95,
        "eyeWideRight": 0.95,
        "eyeSquintLeft": 0.2,
        "eyeSquintRight": 0.2,
        "eyeBlinkLeft": 0.1,
        "eyeBlinkRight": 0.1,
    },

    "sarcastic": {
        "browDownLeft": 0.25,
        "browDownRight": 0.25,
        "browInnerUp": 0.15,
        "eyeSquintLeft": 0.5,
        "eyeSquintRight": 0.5,
        "mouthSmileLeft": 0.35,
        "mouthSmileRight": 0.35,
        "mouthFrownLeft": 0.25,
        "mouthFrownRight": 0.25,
    },

    "thinking": {
        "browDownLeft": 0.35,
        "browDownRight": 0.15,
        "browInnerUp": 0.3,
        "browOuterUpLeft": 0.1,
        "eyeSquintLeft": 0.3,
        "eyeSquintRight": 0.15,
        "mouthSmileLeft": 0.1,
        "mouthSmileRight": 0.05,
    },
}

UPPER_FACE_KEYS = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
]


def get_blendshapes(emotion: str, intensity: float = 1.0) -> dict:
    """
    Returns the blendshape values for the given emotion, scaled by intensity.
    Falls back to neutral if emotion is unknown.
    Only upper-face keys are present.
    For "neutral" we always return the full natural rest pose (intensity is ignored).
    """
    emotion = (emotion or "neutral").lower().strip()
    if emotion == "neutral":
        return NEUTRAL_REST.copy()

    base = EMOTION_MAP.get(emotion, EMOTION_MAP["neutral"])
    if intensity <= 0:
        intensity = 0.0
    return {k: round(v * intensity, 4) for k, v in base.items()}


if __name__ == "__main__":
    # Quick self-test
    print("Available emotions:", list(EMOTION_MAP.keys()))
    print("Happy @ 0.8:", get_blendshapes("happy", 0.8))
    print("Unknown -> neutral:", get_blendshapes("excited", 1.0))
