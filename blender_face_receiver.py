"""
blender_face_receiver.py

Run this INSIDE Blender (Text Editor > open this file > Run Script),
with mediapipe_face_capture.py running separately on your system Python.

It listens on a UDP socket for blendshape JSON packets and drives the
matching shape keys on your Faceit-rigged mesh in real time.

NEW BEHAVIOR:
- Auto-detects the best mesh if TARGET_MESH_NAME is empty / not found.
- When you start the stream (bpy.ops.face.stream_receiver()), it waits up
  to 15 seconds for the first packet from the sender.
- If no signal arrives → it auto-closes.
- Once receiving data, it automatically stops when the sender stops
  sending (no packets for a few seconds).

Setup:
    1. (Optional) Set TARGET_MESH_NAME. Leave as "" or None for auto-detect.
       The script will search for the mesh with the most ARKit-style shape keys
       and will also SELECT it in the 3D View.
    2. Run this script once (registers the operators).
    3. In the Python console or another text block, run:
           bpy.ops.face.stream_receiver()
       to start listening. It will auto-stop on timeout or sender stop.
       Manual stop: bpy.ops.face.stream_stop()
    4. Start mediapipe_face_capture.py (with testing.mp4 or webcam).

If your shape keys don't match MediaPipe's names exactly, use NAME_OVERRIDES.
"""

import json
import queue
import socket
import threading
import time

import bpy

# ---------------------------------------------------------------------------
# CONFIG -- edit these for your project
# ---------------------------------------------------------------------------
UDP_IP = "127.0.0.1"
UDP_PORT = 9001

# Set to your mesh name, or leave empty / None for AUTO-DETECT + AUTO-SELECT
TARGET_MESH_NAME = ""  # e.g. "CC_Base_Body" or "grp_blendShapes_01"

# 0.0 = no smoothing (raw, jittery, but shows true rig quality)
# 0.5-0.8 = increasingly smoothed / laggier
SMOOTHING = 0.4

# Only fill this in for shape keys whose names don't match MediaPipe's
# ARKit-style names exactly, e.g.:
#   NAME_OVERRIDES = {"eyeBlinkLeft": "Eye_Blink_L"}
NAME_OVERRIDES = {}

# Signal detection timeouts (in seconds)
FIRST_SIGNAL_TIMEOUT = 15.0     # Wait this long for the first packet after starting
SENDER_SILENCE_TIMEOUT = 3.5    # If no new data for this long after receiving, stop
# ---------------------------------------------------------------------------

_data_queue: "queue.Queue" = queue.Queue()
_smoothed_values = {}
_running = False
_sock = None
_listener_thread = None

_start_time = None
_last_packet_time = None

# Common ARKit blendshape names used by MediaPipe for scoring auto-detect
_ARKIT_KEYS = {
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
}


def _find_best_face_mesh():
    """Scan the scene for the mesh with the highest number of matching ARKit shape keys."""
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
    return best if best_score >= 3 else None   # require at least a few matches


def _get_target_object():
    """Return the target mesh object (respect configured name, else auto-detect)."""
    global TARGET_MESH_NAME
    if TARGET_MESH_NAME:
        obj = bpy.data.objects.get(TARGET_MESH_NAME)
        if obj and obj.data.shape_keys:
            return obj
    # Auto-detect
    obj = _find_best_face_mesh()
    if obj:
        TARGET_MESH_NAME = obj.name
        return obj
    return None


def _select_mesh(obj):
    """Select the mesh in the viewport and make it active (best effort)."""
    if obj is None:
        return
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    except Exception:
        pass   # safe in case of wrong context


def _listener_loop():
    """Runs on a background thread so the UDP recv never blocks Blender's UI."""
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
    _sock.close()


def _resolve_shape_key_name(mp_name, key_blocks):
    if mp_name in NAME_OVERRIDES:
        override = NAME_OVERRIDES[mp_name]
        return override if override in key_blocks else None
    if mp_name in key_blocks:
        return mp_name
    return None


def _print_shape_keys():
    configured = TARGET_MESH_NAME
    obj = _get_target_object()
    if obj is None:
        print("[face_receiver] No suitable mesh found automatically.")
        print("[face_receiver] Set TARGET_MESH_NAME manually to your Faceit mesh object name.")
        return
    # Determine if we performed auto-detection
    used_auto = (configured in (None, "", "None")) or (configured != obj.name)
    tag = "(auto-detected + selected)" if used_auto else ""
    print(f"[face_receiver] Target mesh: {obj.name} {tag}".strip())
    if obj.data.shape_keys is None:
        print("[face_receiver]   Mesh has no shape keys.")
        return
    print("[face_receiver] Available shape keys:")
    for kb in obj.data.shape_keys.key_blocks:
        print("   -", kb.name)


class FACE_OT_stream_receiver(bpy.types.Operator):
    """Start listening for MediaPipe blendshape data and drive the rig."""

    bl_idname = "face.stream_receiver"
    bl_label = "Start MediaPipe Face Stream"

    _timer = None

    def modal(self, context, event):
        global _last_packet_time

        if not _running:
            return self.cancel(context)

        if event.type == "TIMER":
            now = time.time()

            obj = _get_target_object()
            if obj is None or obj.data.shape_keys is None:
                self.report({"ERROR"}, "Target mesh not found or has no shape keys.")
                return self.cancel(context)

            key_blocks = obj.data.shape_keys.key_blocks

            # Drain the queue, only apply the most recent packet per tick
            latest = None
            while not _data_queue.empty():
                latest = _data_queue.get_nowait()

            if latest is not None:
                _last_packet_time = now
                for mp_name, raw_value in latest.get("blendshapes", {}).items():
                    key_name = _resolve_shape_key_name(mp_name, key_blocks)
                    if key_name is None:
                        continue
                    prev = _smoothed_values.get(mp_name, 0.0)
                    smoothed = prev * SMOOTHING + raw_value * (1.0 - SMOOTHING)
                    _smoothed_values[mp_name] = smoothed
                    key_blocks[key_name].value = smoothed

            # --- Auto timeout logic ---
            if _last_packet_time is None:
                # Haven't received anything yet
                if _start_time is not None and (now - _start_time) > FIRST_SIGNAL_TIMEOUT:
                    self.report({"WARNING"}, f"No sender signal received in {FIRST_SIGNAL_TIMEOUT:.0f}s. Closing.")
                    return self.cancel(context)
            else:
                # We have received data before — stop if sender goes silent
                if (now - _last_packet_time) > SENDER_SILENCE_TIMEOUT:
                    self.report({"INFO"}, "Sender stopped sending. Auto-stopping receiver.")
                    return self.cancel(context)

        return {"PASS_THROUGH"}

    def execute(self, context):
        global _running, _listener_thread, _start_time, _last_packet_time, _smoothed_values

        obj = _get_target_object()
        _print_shape_keys()

        if obj is None:
            self.report({"ERROR"}, "No target mesh found. Set TARGET_MESH_NAME or ensure your mesh has ARKit shape keys.")
            return {"CANCELLED"}

        # Select the mesh in the viewport
        _select_mesh(obj)

        # Reset state
        _smoothed_values.clear()
        _start_time = time.time()
        _last_packet_time = None

        _running = True
        _listener_thread = threading.Thread(target=_listener_loop, daemon=True)
        _listener_thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 30.0, window=context.window)
        wm.modal_handler_add(self)
        self.report({"INFO"}, f"Listening on {UDP_IP}:{UDP_PORT}  (waiting up to {FIRST_SIGNAL_TIMEOUT:.0f}s for sender...)")
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
    """Stop the MediaPipe face stream listener."""

    bl_idname = "face.stream_stop"
    bl_label = "Stop MediaPipe Face Stream"

    def execute(self, context):
        global _running
        _running = False
        self.report({"INFO"}, "Stopped listening.")
        return {"FINISHED"}


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
    print("[face_receiver] Start:  bpy.ops.face.stream_receiver()")
    print("[face_receiver]         (will wait 15s for sender, then auto-stop when sender stops)")
    print("[face_receiver] Stop:   bpy.ops.face.stream_stop()")
