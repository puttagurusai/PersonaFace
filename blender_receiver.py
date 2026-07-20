"""
blender_receiver.py  (Updated for emotion + viseme + live MediaPipe)

Supports two modes simultaneously:
1. Live MediaPipe face capture (from mediapipe_face_capture.py)
2. Orchestrator-driven emotion + rhubarb lip-sync (from orchestrator.py)

Packet formats supported:
- Legacy / MediaPipe: {"blendshapes": {...}} or {"t": ..., "blendshapes": {...}}
- Emotion:             {"type": "emotion", "blendshapes": {...}}   → upper face only
- Viseme:              {"type": "viseme",  "blendshapes": {...}}   → mouth/jaw only

Run inside Blender:
    1. Open this file in Text Editor → Run Script
    2. In Python Console:
         bpy.ops.face.stream_receiver()

Stop with:
    bpy.ops.face.stream_stop()
"""

import json
import queue
import socket
import threading
import time

import bpy

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9001

# Leave empty for AUTO-DETECT + AUTO-SELECT best mesh with shape keys
TARGET_MESH_NAME = ""

# Per-packet-type smoothing (fraction of new sample mixed into target each packet)
# Mouth must track audio tightly or lips look late vs speech.
VISEME_SMOOTHING = 0.92   # nearly snap lips to packet (sync with audio)
EMOTION_SMOOTHING = 0.28  # brows/eyes can ease
SMOOTHING = 0.30

# Glide rates toward TARGET_VALUES each timer tick (30 Hz). Higher = snappier.
VISEME_GLIDE = 0.80       # lips follow targets immediately
EMOTION_GLIDE = 0.28

# Verbose packet logging kills real-time performance (30 prints/sec)
VERBOSE_PACKETS = False

# Only used for legacy/manual name fixing
NAME_OVERRIDES = {}

# Timeouts for orchestrator / live sender
FIRST_SIGNAL_TIMEOUT = 120.0   # How long to wait for the VERY FIRST packet after starting the receiver (the "first signal")

# After the first packet has been received ("connected"), we no longer auto-stop on short silence.
# The old 3.5s SENDER_SILENCE_TIMEOUT was causing unwanted stops during pauses between sentences
# or while user is pasting the next JSON. Once connected, the receiver should stay alive
# until the user explicitly calls bpy.ops.face.stream_stop() (or the whole Blender session ends).
# We keep a very long "post-connection" timeout only as a safety net.
POST_CONNECTION_SILENCE_TIMEOUT = 300.0   # 5 minutes of total silence after connection before giving up

# ---------------------------------------------------------------------------
# UPPER vs LOWER FACE SEPARATION
# ---------------------------------------------------------------------------
# Upper face driven by emotion packets. Mouth smiles/frowns stay available
# to lip-sync (wav2arkit needs stretch/smile for speech shapes).
UPPER_FACE_KEYS = {
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # mouthSmile / mouthFrown intentionally NOT here — lip-sync owns them while speaking
}

# Everything else (jaw, full mouth shapes, etc.) is considered lower/mouth for visemes
# We don't need an explicit list — we just avoid upper keys for viseme packets.

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
_data_queue: "queue.Queue" = queue.Queue()
_smoothed_values = {}
TARGET_VALUES = {}
_tracked_keys = set()
_running = False
_sock = None
_listener_thread = None

_start_time = None
_last_packet_time = None

# For blink logic (Step B)
MEDIAPIPE_ACTIVE = False
_current_emotion = "neutral"
_last_mediapipe_time = 0.0
_blink_queue: "queue.Queue" = queue.Queue()
_blink_thread = None
_eye_gaze_thread = None
_head_thread = None

# For Step F and G
IS_SPEAKING = False
_current_head_pitch = 0.0
_current_head_yaw = 0.0
_current_head_roll = 0.0


# ---------------------------------------------------------------------------
# AUTO MESH DETECTION (kept from previous version)
# ---------------------------------------------------------------------------
_ARKIT_KEYS = {
    "eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
    "browDownLeft", "browDownRight", "browInnerUp",
}

def _find_best_face_mesh():
    best = None
    best_score = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data.shape_keys is None:
            continue
        key_names = {kb.name for kb in obj.data.shape_keys.key_blocks}
        score = len(_ARKIT_KEYS & key_names)
        if score > best_score:
            best_score = score
            best = obj
    return best if best_score >= 2 else None

def _get_target_object():
    global TARGET_MESH_NAME
    if TARGET_MESH_NAME:
        obj = bpy.data.objects.get(TARGET_MESH_NAME)
        if obj and obj.data.shape_keys:
            return obj
    obj = _find_best_face_mesh()
    if obj:
        TARGET_MESH_NAME = obj.name
        return obj
    return None

def _select_mesh(obj):
    if obj is None:
        return
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    except Exception:
        pass

# ---------------------------------------------------------------------------
# LISTENER
# ---------------------------------------------------------------------------
def _listener_loop():
    global _sock
    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _sock.bind((UDP_IP, UDP_PORT))
    _sock.settimeout(0.5)
    while _running:
        try:
            data, _ = _sock.recvfrom(65536)
            _data_queue.put(json.loads(data.decode("utf-8")))
        except socket.timeout:
            continue
        except OSError:
            break
    if _sock:
        _sock.close()


def _get_blink_interval():
    """Return random blink interval in seconds based on current emotion."""
    import random
    emotion = _current_emotion.lower()
    if emotion == "fearful":
        return random.uniform(1.5, 3.0)
    elif emotion == "angry":
        return random.uniform(2.0, 3.5)
    elif emotion == "thinking":
        return random.uniform(5.0, 9.0)
    elif emotion == "sad":
        return random.uniform(4.0, 8.0)
    elif emotion == "happy":
        return random.uniform(3.5, 7.0)
    else:  # neutral or others
        return random.uniform(3.0, 6.0)


def _blink_loop():
    """Background blink thread. Ramps eyeBlink keys directly."""
    import random
    import time as _time
    while _running:
        interval = _get_blink_interval()
        _time.sleep(interval)

        if not _running:
            break
        if MEDIAPIPE_ACTIVE and (time.time() - _last_mediapipe_time < 2.0):
            continue  # pause blinks if live capture was active recently

        # Ramp up
        for i in range(5):  # 80ms / 20ms = 4 steps, +1 for 0
            val = (i + 1) / 5.0   # 0.2, 0.4, 0.6, 0.8, 1.0
            _blink_queue.put({"eyeBlinkLeft": val, "eyeBlinkRight": val})
            _time.sleep(0.02)

        _time.sleep(0.04)  # hold at 1.0 for 40ms

        # Ramp down
        for i in range(6):  # 120ms / 20ms = 6 steps
            val = 1.0 - (i / 6.0)   # 1.0 -> 0.0
            _blink_queue.put({"eyeBlinkLeft": val, "eyeBlinkRight": val})
            _time.sleep(0.02)


def eye_gaze_loop():
    """Background thread for eye gaze and micro saccades (Step F). Writes to TARGET_VALUES for gliding."""
    import random
    import time as _time
    gaze_targets = {
        "center": (0.0, 0.0),
        "left": (-0.12, 0.0),
        "right": (0.12, 0.0),
        "up": (0.0, 0.12),
        "down": (0.0, -0.08),
    }
    current_x, current_y = 0.0, 0.0
    last_change = _time.time()
    last_saccade = _time.time()

    while _running:
        now = _time.time()
        is_speaking = IS_SPEAKING

        if is_speaking:
            max_val = 0.20
            change_interval = random.uniform(1.0, 2.5)
        else:
            max_val = 0.15
            change_interval = random.uniform(2.0, 5.0)

        # Saccades - rapid micro flicks
        if now - last_saccade > random.uniform(1.5, 3.0):
            sacc_x = random.uniform(-0.1, 0.1) * max_val
            sacc_y = random.uniform(-0.08, 0.08) * max_val
            # Quick move
            for _ in range(2):
                current_x = current_x * 0.6 + sacc_x * 0.4
                current_y = current_y * 0.6 + sacc_y * 0.4
                # Update TARGET_VALUES (will be glided in main timer)
                TARGET_VALUES["eyeLookInLeft"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutLeft"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpLeft"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownLeft"] = max(0.0, -current_y) if current_y < 0 else 0.0
                TARGET_VALUES["eyeLookInRight"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutRight"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpRight"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownRight"] = max(0.0, -current_y) if current_y < 0 else 0.0
                _tracked_keys.update([
                    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
                    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"
                ])
                _time.sleep(0.02)
            # Return to previous
            for _ in range(3):
                current_x *= 0.7
                current_y *= 0.7
                TARGET_VALUES["eyeLookInLeft"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutLeft"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpLeft"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownLeft"] = max(0.0, -current_y) if current_y < 0 else 0.0
                TARGET_VALUES["eyeLookInRight"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutRight"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpRight"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownRight"] = max(0.0, -current_y) if current_y < 0 else 0.0
                _tracked_keys.update([
                    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
                    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"
                ])
                _time.sleep(0.02)
            last_saccade = now

        # Main gaze drift
        if now - last_change > change_interval:
            target_name = random.choice(list(gaze_targets.keys()))
            target_x, target_y = gaze_targets[target_name]
            target_x *= max_val / 0.15
            target_y *= max_val / 0.15

            steps = random.randint(10, 20)
            step_time = random.uniform(0.2, 0.4) / steps
            for s in range(steps):
                frac = (s + 1) / steps
                current_x = current_x * (1 - frac) + target_x * frac
                current_y = current_y * (1 - frac) + target_y * frac
                TARGET_VALUES["eyeLookInLeft"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutLeft"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpLeft"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownLeft"] = max(0.0, -current_y) if current_y < 0 else 0.0
                TARGET_VALUES["eyeLookInRight"] = max(0.0, -current_x) if current_x < 0 else 0.0
                TARGET_VALUES["eyeLookOutRight"] = max(0.0, current_x) if current_x > 0 else 0.0
                TARGET_VALUES["eyeLookUpRight"] = max(0.0, current_y) if current_y > 0 else 0.0
                TARGET_VALUES["eyeLookDownRight"] = max(0.0, -current_y) if current_y < 0 else 0.0
                _tracked_keys.update([
                    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
                    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight"
                ])
                _time.sleep(step_time)
            last_change = now

        if int(now) % 5 == 0:
            print(f"[eye_gaze_loop] current_x={current_x:.3f} current_y={current_y:.3f} speaking={is_speaking}")

        _time.sleep(0.1)


def head_movement_loop():
    """Background thread for subtle head movement (Step G). Uses blendshapes for tilt simulation + sends 'head' packet for rotation."""
    import random
    import time as _time
    import math

    last_nod = 0.0
    last_drift = 0.0
    current_pitch = 0.0
    current_yaw = 0.0
    current_roll = 0.0
    prev_speaking = False

    while _running:
        now = _time.time()
        is_speaking = IS_SPEAKING
        emotion = _current_emotion.lower()

        # Breathing always (subtle)
        breath = math.sin(now * 0.8) * 0.02
        target_pitch = breath
        target_yaw = current_yaw
        target_roll = current_roll

        if is_speaking:
            # Nod on sentence start
            if not prev_speaking:
                target_pitch += 0.12
                last_nod = now
            prev_speaking = True
            # Drift
            if now - last_drift > random.uniform(1.0, 2.5):
                target_yaw = random.uniform(-0.08, 0.08)
                last_drift = now
            # Emotion specific
            if emotion == "surprised":
                target_pitch -= 0.10  # pull back
            elif emotion == "angry":
                target_pitch += 0.08  # lean in
            elif emotion == "thinking":
                target_roll = 0.07  # tilt
            elif emotion == "sad":
                target_pitch += 0.10  # drop
        else:
            prev_speaking = False
            # Idle slow drift
            if now - last_drift > random.uniform(4.0, 8.0):
                target_pitch = random.uniform(-0.08, 0.08) + breath
                target_yaw = random.uniform(-0.10, 0.10)
                target_roll = random.uniform(-0.05, 0.05)
                last_drift = now

        # Low-pass filter (weight)
        current_pitch = current_pitch * 0.85 + target_pitch * 0.15
        current_yaw = current_yaw * 0.85 + target_yaw * 0.15
        current_roll = current_roll * 0.85 + target_roll * 0.15

        # Update for blendshape tilt simulation (asymmetric cheekSquint + brow for tilt feel)
        TARGET_VALUES["cheekSquintLeft"] = max(0.0, min(1.0, current_roll * -0.5 + 0.02))
        TARGET_VALUES["cheekSquintRight"] = max(0.0, min(1.0, current_roll * 0.5 + 0.02))
        TARGET_VALUES["browDownLeft"] = max(0.0, min(1.0, current_pitch * 0.3 if current_pitch > 0 else 0))
        TARGET_VALUES["browDownRight"] = max(0.0, min(1.0, current_pitch * 0.3 if current_pitch > 0 else 0))

        _tracked_keys.update(["cheekSquintLeft", "cheekSquintRight", "browDownLeft", "browDownRight"])

        # Apply head rotation directly to HEAD_CONTROLLER Empty (local to Blender)
        obj = bpy.data.objects.get("HEAD_CONTROLLER")
        if obj:
            obj.rotation_euler[0] = current_pitch * 0.15  # X pitch
            obj.rotation_euler[1] = current_roll * 0.10   # Y roll
            obj.rotation_euler[2] = current_yaw * 0.18    # Z yaw
        if int(now) % 5 == 0:  # debug every ~5s
            print(f"[head_loop] breathing pitch={current_pitch:.3f} yaw={current_yaw:.3f} roll={current_roll:.3f} speaking={is_speaking}")

        _time.sleep(0.05)  # ~20fps for head (slow movement)


# ---------------------------------------------------------------------------
# PACKET HANDLING
# ---------------------------------------------------------------------------
def _resolve_shape_key_name(mp_name, key_blocks):
    if mp_name in NAME_OVERRIDES:
        override = NAME_OVERRIDES[mp_name]
        return override if override in key_blocks else None
    if mp_name in key_blocks:
        return mp_name
    return None

def _apply_blendshapes(obj, blendshapes_dict, allowed_keys=None, smoothing=None):
    """Update TARGET_VALUES from incoming packet (don't apply to mesh instantly).
    Use the provided smoothing to blend into target.
    If smoothing is None, uses EMOTION_SMOOTHING.
    """
    if obj is None or obj.data.shape_keys is None:
        return

    if smoothing is None:
        smoothing = EMOTION_SMOOTHING

    key_blocks = obj.data.shape_keys.key_blocks

    for mp_name, raw_value in blendshapes_dict.items():
        if allowed_keys is not None and mp_name not in allowed_keys:
            continue

        key_name = _resolve_shape_key_name(mp_name, key_blocks)
        if key_name is None:
            continue

        old_target = TARGET_VALUES.get(key_name, 0.0)
        new_target = old_target * (1.0 - smoothing) + raw_value * smoothing
        TARGET_VALUES[key_name] = new_target
        _tracked_keys.add(key_name)

def _handle_packet(packet):
    """Route packet to the correct face region based on type."""
    if not isinstance(packet, dict):
        return

    blendshapes = packet.get("blendshapes", packet)  # support legacy flat or wrapped

    obj = _get_target_object()
    if obj is None:
        return
    if getattr(obj, 'data', None) is None or getattr(obj.data, 'shape_keys', None) is None:
        return
    key_blocks = obj.data.shape_keys.key_blocks

    ptype = packet.get("type")

    if VERBOSE_PACKETS:
        print(f"[face_receiver] Received packet type={ptype}")

    global IS_SPEAKING

    if ptype == "emotion":
        # Upper face only (brows, eyes, cheeks)
        global _current_emotion
        _current_emotion = packet.get("emotion", _current_emotion)
        if VERBOSE_PACKETS:
            print(f"[face_receiver] EMOTION keys={list(blendshapes.keys())} ({_current_emotion})")
        _apply_blendshapes(obj, blendshapes, allowed_keys=UPPER_FACE_KEYS, smoothing=EMOTION_SMOOTHING)

    elif ptype == "viseme":
        # Mouth / jaw — set targets directly so lips stay in sync with audio
        # (no lag from heavy smoothing; mesh still uses light VISEME_GLIDE)
        IS_SPEAKING = True
        lower_only = {k: v for k, v in blendshapes.items() if k not in UPPER_FACE_KEYS}
        if not lower_only:
            return
        for mp_name, raw_value in lower_only.items():
            key_name = _resolve_shape_key_name(mp_name, key_blocks)
            if key_name is None:
                continue
            v = float(raw_value)
            TARGET_VALUES[key_name] = v
            _tracked_keys.add(key_name)
            # Seed smoothed value close to target so first frames aren't stuck at 0
            prev = _smoothed_values.get(key_name, v)
            _smoothed_values[key_name] = prev * (1.0 - VISEME_SMOOTHING) + v * VISEME_SMOOTHING

    elif ptype == "rest_pose":
        # Full-face rest. smooth=True (end of sentence) only updates targets so the
        # existing glide system eases to neutral — no sudden jerk. smooth=False
        # (startup / force) snaps immediately.
        IS_SPEAKING = False
        smooth = bool(packet.get("smooth", False))
        if smooth:
            print(f"[face_receiver] Applying REST_POSE (smooth) — gliding face to neutral rest")
            for k, v in blendshapes.items():
                key_name = _resolve_shape_key_name(k, key_blocks)
                if key_name:
                    TARGET_VALUES[key_name] = v
                    _tracked_keys.add(key_name)
            print("[face_receiver] Neutral rest targets set; glide will settle face smoothly")
        else:
            print(f"[face_receiver] Applying REST_POSE (instant) — forcing full face to neutral rest")
            for k, v in blendshapes.items():
                key_name = _resolve_shape_key_name(k, key_blocks)
                if key_name:
                    TARGET_VALUES[key_name] = v
                    _smoothed_values[key_name] = v
                    _tracked_keys.add(key_name)
                    if key_name in key_blocks:
                        key_blocks[key_name].value = v
            print("[face_receiver] Face forced to NEUTRAL_REST (jaw slightly open, tiny eye squint, etc.)")

    elif ptype == "head":
        # Head rotation packet (Step G)
        pitch = packet.get("pitch", 0.0)
        yaw = packet.get("yaw", 0.0)
        roll = packet.get("roll", 0.0)
        obj = bpy.data.objects.get("HEAD_CONTROLLER")
        if obj:
            obj.rotation_euler[0] = pitch * 0.15  # X
            obj.rotation_euler[1] = roll * 0.10   # Y
            obj.rotation_euler[2] = yaw * 0.18    # Z
        # Also update for blendshape tilt if no controller
        TARGET_VALUES["cheekSquintLeft"] = max(0.0, min(1.0, roll * -0.5))
        TARGET_VALUES["cheekSquintRight"] = max(0.0, min(1.0, roll * 0.5))
        _tracked_keys.update(["cheekSquintLeft", "cheekSquintRight"])

    else:
        # Legacy / live MediaPipe capture — apply everything (full face)
        global MEDIAPIPE_ACTIVE, _last_mediapipe_time
        _last_mediapipe_time = time.time()
        MEDIAPIPE_ACTIVE = True
        _apply_blendshapes(obj, blendshapes, allowed_keys=None)

# ---------------------------------------------------------------------------
# PRINT SHAPE KEYS
# ---------------------------------------------------------------------------
def _print_shape_keys():
    configured = TARGET_MESH_NAME
    obj = _get_target_object()
    if obj is None:
        print("[face_receiver] No suitable mesh found automatically.")
        print("[face_receiver] Set TARGET_MESH_NAME manually.")
        return
    used_auto = (not configured) or (configured != obj.name)
    tag = "(auto-detected + selected)" if used_auto else ""
    print(f"[face_receiver] Target mesh: {obj.name} {tag}".strip())

    if obj.data.shape_keys is None:
        print("[face_receiver]   Mesh has no shape keys.")
        return
    print("[face_receiver] Available shape keys:")
    for kb in obj.data.shape_keys.key_blocks:
        print("   -", kb.name)

# ---------------------------------------------------------------------------
# OPERATORS
# ---------------------------------------------------------------------------
class FACE_OT_stream_receiver(bpy.types.Operator):
    bl_idname = "face.stream_receiver"
    bl_label = "Start MediaPipe / Orchestrator Face Stream"

    _timer = None

    def modal(self, context, event):
        global _last_packet_time

        if not _running:
            return self.cancel(context)

        if event.type == "TIMER":
            now = time.time()

            # Drain queue — process EVERY packet (do not keep only the last).
            # Multi-agent orchestrator sends viseme + emotion + head each frame;
            # dropping all but the last would leave only "head" and lips never move.
            got_any = False
            while not _data_queue.empty():
                try:
                    pkt = _data_queue.get_nowait()
                except Exception:
                    break
                if _last_packet_time is None:
                    print("[face_receiver] *** CONNECTED to orchestrator! ***")
                    print("[face_receiver] Receiving emotion/viseme/head packets (all types applied).")
                _last_packet_time = now
                got_any = True
                try:
                    _handle_packet(pkt)
                except Exception as e:
                    print(f"[face_receiver] Error handling packet: {e}")

            # Glide mesh toward targets. Mouth uses faster VISEME_GLIDE so lips open fully.
            try:
                obj = _get_target_object()
                if obj and obj.data.shape_keys:
                    key_blocks = obj.data.shape_keys.key_blocks
                    for key_name in list(_tracked_keys):
                        target = TARGET_VALUES.get(key_name, 0.0)
                        current = _smoothed_values.get(key_name, 0.0)
                        is_upper = key_name in UPPER_FACE_KEYS or key_name.startswith("eye") or key_name.startswith("brow")
                        rate = EMOTION_GLIDE if is_upper else VISEME_GLIDE
                        new_val = current * (1.0 - rate) + target * rate
                        _smoothed_values[key_name] = new_val
                        if key_name in key_blocks:
                            key_blocks[key_name].value = new_val
            except Exception as e:
                print(f"[face_receiver] Error in gliding: {e}")

            # Reset mediapipe active flag if no recent legacy packets
            global MEDIAPIPE_ACTIVE
            if time.time() - _last_mediapipe_time > 2.0:
                MEDIAPIPE_ACTIVE = False

            # Apply any pending blink values directly (bypass normal queue for procedural blinks)
            obj = _get_target_object()
            if obj and obj.data.shape_keys:
                key_blocks = obj.data.shape_keys.key_blocks
                while not _blink_queue.empty():
                    try:
                        blink_dict = _blink_queue.get_nowait()
                        for k, v in blink_dict.items():
                            if k in key_blocks:
                                key_blocks[k].value = v
                    except Exception:
                        break

            # Auto-timeout logic explained:
            #
            # === FIRST SIGNAL ===
            # _last_packet_time starts as None.
            # The very first packet the receiver ever sees is the "first signal".
            # We only care about how long since we *started* the operator (_start_time).
            # If no packet arrives within FIRST_SIGNAL_TIMEOUT (120s), we stop.
            # This protects you if you start the receiver but forget to start the sender.
            #
            # When the first packet arrives:
            #   - We print "*** CONNECTED to orchestrator! ***"
            #   - We set _last_packet_time = now
            #   - From now on we are in "connected" mode.
            #
            # === ONGOING / SECOND AND LATER SIGNALS ===
            # After the first signal, every new packet is a "subsequent signal".
            # Each one updates _last_packet_time.
            # If more than POST_CONNECTION_SILENCE_TIMEOUT passes with *no packets at all*,
            # we assume the sender has permanently stopped and we auto-cancel.
            #
            # Important: once connected, we do NOT stop after short gaps anymore.
            # We wait for a very long silence (or explicit bpy.ops.face.stream_stop()).
            # This is what you wanted: "wait till exit of sender once connected".
            if _last_packet_time is None:
                # Still waiting for the very first packet ("first signal")
                if _start_time is not None and (now - _start_time) > FIRST_SIGNAL_TIMEOUT:
                    self.report({"WARNING"}, f"No signal received in {FIRST_SIGNAL_TIMEOUT:.0f}s. Stopping.")
                    return self.cancel(context)
            else:
                # Connected — only stop after very long total silence since last packet
                if (now - _last_packet_time) > POST_CONNECTION_SILENCE_TIMEOUT:
                    self.report({"INFO"}, "No data for a very long time after connection. Auto-stopping for safety.")
                    return self.cancel(context)

        return {"PASS_THROUGH"}

    def execute(self, context):
        global _running, _listener_thread, _start_time, _last_packet_time, _smoothed_values

        obj = _get_target_object()
        _print_shape_keys()

        if obj is None:
            self.report({"ERROR"}, "No target mesh with shape keys found.")
            return {"CANCELLED"}

        _select_mesh(obj)

        _smoothed_values.clear()
        _start_time = time.time()
        _last_packet_time = None

        _running = True
        _listener_thread = threading.Thread(target=_listener_loop, daemon=True)
        _listener_thread.start()

        global _blink_thread
        _blink_thread = threading.Thread(target=_blink_loop, daemon=True)
        _blink_thread.start()

        global _eye_gaze_thread, _head_thread
        _eye_gaze_thread = threading.Thread(target=eye_gaze_loop, daemon=True)
        _eye_gaze_thread.start()
        print("[face_receiver] eye_gaze_loop thread started")

        _head_thread = threading.Thread(target=head_movement_loop, daemon=True)
        _head_thread.start()
        print("[face_receiver] head_movement_loop thread started (breathing + head movement active)")

        print("[face_receiver] All background threads (blink, gaze, head) started. Check console for periodic [head_loop] and [eye_gaze_loop] messages.")

        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 30.0, window=context.window)
        wm.modal_handler_add(self)
        self.report({"INFO"}, f"Listening on {UDP_IP}:{UDP_PORT} (will stay connected after first packet; use stream_stop() to exit)")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        global _running
        _running = False
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        return {"CANCELLED"}


class FACE_OT_stream_stop(bpy.types.Operator):
    bl_idname = "face.stream_stop"
    bl_label = "Stop Face Stream"

    def execute(self, context):
        global _running
        _running = False
        self.report({"INFO"}, "Stopped listening.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------------
_classes = (FACE_OT_stream_receiver, FACE_OT_stream_stop)

def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

def unregister():
    global _running
    _running = False
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
    _print_shape_keys()
    print("[face_receiver] Ready.")
    print("Start: bpy.ops.face.stream_receiver()")
    print("Stop:  bpy.ops.face.stream_stop()")
    print("Supports: live MediaPipe + orchestrator emotion/viseme packets")
