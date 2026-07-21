"""
orchestrator_brain.py

Brain + Parler Mini pipeline (does NOT gut orchestrator.py).

  JSON → Parler TTS → WAV
       → emotion_manager (emotion_26d)
       → Brain (upper face / expression)  +  wav2arkit (mouth lip-sync)   [default hybrid]
       → play_brain_output @ 30fps (viseme lower + emotion upper UDP)
       → blender_receiver.py

Lip modes (MOUTH_ENGINE):
  hybrid   — wav2arkit mouth (good lip-sync) + Brain upper + emotion_map  [DEFAULT]
  brain    — pure Brain mouth (softer / flatter; research mode)
  wav2arkit — wav2arkit mouth only + emotion_map upper (classic look)

Why hybrid: pure Brain lower-head motion is low-variance (jaw std ~0.01 vs
wav2arkit ~0.04) so lips look open but not speech-locked — “Rhubarb-like”.
wav2arkit was trained for continuous audio→ARKit lips; use that for mouth.

Run:
  1. Blender: blender_receiver.py → bpy.ops.face.stream_receiver()
  2. python orchestrator_brain.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

# Reuse shared config / UDP / TTS / JSON input from classic orchestrator
import orchestrator as base
import emotion_map
import brain_inference
import prosody_gpu
import wav2arkit
from emotion_manager import get_emotion_manager
from parler_voice import load_parler, build_voice_style, generate_speech

# Hybrid is the production path (Brain still runs for upper / emotion_26d)
base.LIPSYNC_ENGINE = "brain"
base.BRAIN_ALLOW_STUB = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEMP_DIR = base.TEMP_DIR
TEMP_DIR.mkdir(exist_ok=True)

# NVIDIA ARKit order (same as brain_inference)
NVIDIA_ARKIT_ORDER = brain_inference.NVIDIA_ARKIT_ORDER

# Match brain_model ARKIT_LOWER / ARKIT_UPPER (scatter layout) + blender_receiver.
# Lower (viseme): cheekPuff + jaw + full mouth + tongue  — NOT nose
# Upper (emotion): brows, cheek squints, eyes, nose — NOT mouth smiles (lips stay on viseme)
from brain_model import ARKIT_LOWER_INDICES, ARKIT_UPPER_INDICES

LOWER_FACE_KEYS = {NVIDIA_ARKIT_ORDER[i] for i in ARKIT_LOWER_INDICES}
# Mouth keys owned by lip-sync engine (must match wav2arkit.MOUTH_JAW_KEYS spirit)
MOUTH_LIPSYNC_KEYS = set(wav2arkit.MOUTH_JAW_KEYS)

# blender_receiver only applies a subset of upper keys on emotion packets
BLENDER_UPPER_KEYS = set(emotion_map.UPPER_FACE_KEYS) & {
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
}
# Full Brain upper scatter set (for logging); playback uses BLENDER_UPPER_KEYS
UPPER_FACE_KEYS = {NVIDIA_ARKIT_ORDER[i] for i in ARKIT_UPPER_INDICES}

# How strongly to pull Brain upper toward emotion_map presets (wav2arkit path used presets only)
EMOTION_MAP_BLEND = 0.55

# hybrid | brain | wav2arkit  (env MOUTH_ENGINE overrides)
MOUTH_ENGINE = os.environ.get("MOUTH_ENGINE", "hybrid").strip().lower()
if MOUTH_ENGINE not in ("hybrid", "brain", "wav2arkit"):
    MOUTH_ENGINE = "hybrid"

# A/V sync — lips were slightly AHEAD of heard audio (common on Windows).
# Total hold-back = LIP_DELAY_MS + LIP_MODEL_LEAD_MS.
#   LIP_DELAY_MS      = audio device / buffer latency compensation
#   LIP_MODEL_LEAD_MS = wav2arkit often opens mouth slightly before the phoneme
# Env: $env:LIP_DELAY_MS=200 ; $env:LIP_MODEL_LEAD_MS=40
# Defaults tuned for Windows WASAPI (~120–160ms). Raise if lips still lead.
LIP_DELAY_MS = float(os.environ.get("LIP_DELAY_MS", "150"))
LIP_MODEL_LEAD_MS = float(os.environ.get("LIP_MODEL_LEAD_MS", "30"))

# How long after last speech frame to ease mouth shut before full rest (seconds)
TAIL_EASE_S = float(os.environ.get("LIP_TAIL_EASE_S", "0.12"))


def send_rest_pose(smooth: bool = True) -> None:
    base.send_udp({
        "type": "rest_pose",
        "smooth": smooth,
        "blendshapes": emotion_map.NEUTRAL_REST.copy(),
    })


def _blend_upper(brain_upper: dict, emotion: str, intensity: float) -> dict:
    """
    Merge Brain upper-face channels with emotion_map presets.

    Why: classic wav2arkit path looked expressive because upper came from emotion_map.
    Pure Brain upper is often flatter / wrong scale — blend restores readable emotion
    while keeping Brain micro-motion.
    """
    preset = emotion_map.get_blendshapes(emotion, intensity)
    out = {}
    a = float(EMOTION_MAP_BLEND)
    for k in BLENDER_UPPER_KEYS:
        b = float(brain_upper.get(k, 0.0))
        p = float(preset.get(k, 0.0))
        # Weighted max-bias blend: keep peaks from either source
        out[k] = max(b * (1.0 - a) + p * a, b * 0.85, p * 0.5)
    return out


def _merge_mouth_upper(
    mouth_frames: list[dict],
    upper_frames: list[dict] | None,
    fps: float = 30.0,
) -> list[dict]:
    """
    Build per-frame dicts: mouth keys from mouth_frames, upper from upper_frames.
    Length follows mouth (lip-sync master clock). Upper is time-sampled if lengths differ.
    """
    if not mouth_frames:
        return list(upper_frames or [])

    n = len(mouth_frames)
    out: list[dict] = []
    for i in range(n):
        merged = {k: float(v) for k, v in mouth_frames[i].items() if k in MOUTH_LIPSYNC_KEYS}
        # Prefer dedicated mouth engine values; never carry chronic Brain mouthClose
        if "mouthClose" in merged:
            # wav2arkit already suppresses; keep its value (small). Zero if huge.
            if merged["mouthClose"] > 0.2:
                merged["mouthClose"] = 0.0

        if upper_frames:
            t = i / fps
            u = wav2arkit.frame_at_time(upper_frames, t, fps=fps)
            for k, v in u.items():
                if k in BLENDER_UPPER_KEYS or k in UPPER_FACE_KEYS:
                    if k not in MOUTH_LIPSYNC_KEYS:
                        merged[k] = float(v)
        out.append(merged)
    return out


def _fit_frames_to_audio_duration(
    frames: list[dict],
    audio_duration: float,
    fps: float,
) -> list[dict]:
    """
    Resample lip frames so timeline length == audio length.

    Fixes: lips freeze while trailing audio still plays (model returned fewer
    frames than audio_duration * fps). Time-warps the sequence to span the
    full WAV, then pads a short ease-to-closed tail if needed.
    """
    if not frames or audio_duration <= 0:
        return list(frames or [])

    target_n = max(1, int(round(audio_duration * fps)))
    src_n = len(frames)
    src_dur = max(src_n / fps, 1.0 / fps)

    # If already within ~1 frame of audio length, keep as-is
    if abs(src_n - target_n) <= 1:
        out = list(frames)
    else:
        out = []
        for i in range(target_n):
            # Wall time along full audio; sample source proportionally
            t_audio = i / fps
            t_src = t_audio * (src_dur / audio_duration)
            # Clamp into source span
            t_src = min(max(0.0, t_src), max(0.0, (src_n - 1) / fps))
            out.append(wav2arkit.frame_at_time(frames, t_src, fps=fps))

    # Ensure last ~TAIL_EASE_S seconds ease mouth toward closed (not freeze open)
    ease_n = max(1, int(round(TAIL_EASE_S * fps)))
    if len(out) > ease_n + 2:
        rest = {k: 0.0 for k in MOUTH_LIPSYNC_KEYS}
        # Keep a mild closed-mouth baseline
        rest["jawOpen"] = 0.0
        for j in range(ease_n):
            idx = len(out) - ease_n + j
            frac = (j + 1) / float(ease_n)
            # ease-in cubic
            ease = frac * frac * (3.0 - 2.0 * frac)
            cur = out[idx]
            blended = {}
            keys = set(cur.keys()) | set(rest.keys())
            for k in keys:
                a = float(cur.get(k, 0.0))
                b = float(rest.get(k, 0.0))
                blended[k] = a * (1.0 - ease) + b * ease
            out[idx] = blended

    return out


def _split_viseme_emotion(
    frame_dict: dict,
    emotion: str,
    intensity: float,
) -> tuple[dict, dict]:
    """Build viseme (mouth) + emotion (upper) packets from one frame dict."""
    lower_keys = {
        k: float(v)
        for k, v in frame_dict.items()
        if k in MOUTH_LIPSYNC_KEYS or k in LOWER_FACE_KEYS
    }
    if lower_keys.get("mouthClose", 0.0) > 0.15:
        lower_keys["mouthClose"] = min(lower_keys["mouthClose"], 0.05)
    for k in list(lower_keys.keys()):
        if k in BLENDER_UPPER_KEYS and k not in MOUTH_LIPSYNC_KEYS:
            lower_keys.pop(k, None)

    brain_upper = {k: float(v) for k, v in frame_dict.items() if k in BLENDER_UPPER_KEYS}
    upper_keys = _blend_upper(brain_upper, emotion, intensity)
    return lower_keys, upper_keys


def play_brain_output(
    wav_path: str,
    frames: list[dict],
    device=None,
    emotion: str | None = None,
    intensity: float | None = None,
    fps: float = 30.0,
) -> None:
    """
    Play WAV and stream frames at ~30 fps with *time-based* sampling.

    Sync rules:
      • Fit frames to full audio duration (no frozen mouth while audio continues)
      • LIP_DELAY_MS holds lips back so they match heard audio (not ahead)
      • Keep sending until audio device finishes; rest only after playback ends
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = float(len(audio) / float(sr))
    emo = emotion if emotion is not None else base.current_emotion
    inten = float(intensity if intensity is not None else base.current_intensity)

    if not frames:
        print("  [play_brain] No frames — playing audio only")
        sd.play(audio, sr)
        sd.wait()
        send_rest_pose(smooth=True)
        return

    # Stretch/pad lips to cover entire WAV timeline
    n_before = len(frames)
    frames = _fit_frames_to_audio_duration(frames, duration, fps)
    n_frames = len(frames)
    frame_span = n_frames / float(fps)

    # Total lip hold-back vs wall clock from sd.play()
    lip_hold = max(0.0, (LIP_DELAY_MS + LIP_MODEL_LEAD_MS) / 1000.0)

    frame_interval = 1.0 / float(fps)
    print(
        f"  [play_brain] audio={duration:.2f}s frames={n_before}→{n_frames} "
        f"span={frame_span:.2f}s @ {fps:.0f}fps "
        f"lip_hold={lip_hold*1000:.0f}ms "
        f"(device={LIP_DELAY_MS:.0f}+model={LIP_MODEL_LEAD_MS:.0f}) "
        f"mouth_engine={MOUTH_ENGINE} emotion={emo}"
    )

    # Closed mouth packet (never leak first speech frame during hold-back)
    closed_mouth = {k: 0.0 for k in MOUTH_LIPSYNC_KEYS}
    closed_upper = _blend_upper({}, emo, inten * 0.35)

    def _send_pair(frame_dict: dict, e: str, inten_f: float) -> None:
        lower_keys, upper_keys = _split_viseme_emotion(frame_dict, e, inten_f)
        if lower_keys:
            base.send_udp({"type": "viseme", "blendshapes": lower_keys})
        if upper_keys:
            base.send_udp({
                "type": "emotion",
                "emotion": e,
                "blendshapes": upper_keys,
            })

    # Soft neutral targets (no hard snap that steals first phonemes)
    send_rest_pose(smooth=True)
    _send_pair(closed_mouth, emo, inten * 0.2)

    # --- AUDIO FIRST, LIPS HELD CLOSED, then both advance in lockstep ---
    # sd.play returns before the first sample is heard (device buffer). Holding
    # lips closed for lip_hold ms keeps mouth from starting ahead of sound.
    sd.play(audio.astype(np.float32, copy=False), int(sr), blocking=False)
    audio_start = time.perf_counter()

    # Phase 1: closed mouth while audio buffer fills / lead compensation
    while True:
        wall_t = time.perf_counter() - audio_start
        if wall_t >= lip_hold:
            break
        _send_pair(closed_mouth, emo, inten * 0.25)
        time.sleep(frame_interval)

    # Phase 2: lips track audio content.
    # At this moment audio has been "playing" ~lip_hold seconds (minus residual
    # device lag). We set lip clock so frame t maps to wall (audio_start + lip_hold + t).
    lip_clock0 = time.perf_counter()
    last_frame_time = lip_clock0
    tick = 0
    last_sent: dict = dict(closed_mouth)

    while True:
        now = time.perf_counter()
        # Time since lips were allowed to move
        lip_t = now - lip_clock0
        # Audio content time ≈ time since play minus hold (aligns mouth to speech)
        # Using lip_t directly: when lips unlock, show frame 0 as speech is heard.
        wall_since_play = now - audio_start

        # Stop when past end of audio content (+ small pad)
        if wall_since_play >= duration + 0.03:
            break

        lip_t_clamped = min(max(0.0, lip_t), max(0.0, (n_frames - 1) / fps))
        frame_dict = wav2arkit.frame_at_time(frames, lip_t_clamped, fps=fps)
        last_sent = frame_dict
        _send_pair(frame_dict, emo, inten)

        if tick % 15 == 0:
            print(
                f"  [play_brain] audio_t={wall_since_play:.2f}s lip_t={lip_t:.2f}s "
                f"jawOpen={frame_dict.get('jawOpen', 0):.3f} "
                f"mouthPucker={frame_dict.get('mouthPucker', 0):.3f}"
            )
        tick += 1

        sleep_time = frame_interval - (time.perf_counter() - last_frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_frame_time = time.perf_counter()

    # Audio must finish before rest
    sd.wait()

    # Smooth settle AFTER audio ends only
    for step in range(1, 10):
        frac = step / 9.0
        ease = 1.0 - (1.0 - frac) ** 2
        lower = {
            k: float(last_sent.get(k, 0.0)) * (1.0 - ease)
            for k in MOUTH_LIPSYNC_KEYS
            if k in last_sent and k != "mouthClose"
        }
        brain_upper = {
            k: float(last_sent.get(k, 0.0)) * (1.0 - ease)
            for k in BLENDER_UPPER_KEYS
            if k in last_sent
        }
        upper = _blend_upper(brain_upper, "neutral", 1.0 - ease)
        if lower:
            base.send_udp({"type": "viseme", "blendshapes": lower})
        if upper:
            base.send_udp({"type": "emotion", "emotion": "neutral", "blendshapes": upper})
        time.sleep(0.025)

    send_rest_pose(smooth=True)
    print("  [play_brain] Done (rest after audio finished).")


def process_sentence_brain(sentence: dict, idx: int) -> None:
    """
    Full sentence:
      1) Parler TTS
      2) Mouth: wav2arkit (hybrid/default) and/or Brain
      3) Upper: Brain + emotion_map blend
      4) play_brain_output (dual UDP + audio)
    """
    text = sentence["text"]
    emotion = sentence["emotion"]
    intensity = float(sentence["intensity"])

    print(f"\n{'=' * 50}")
    print(f"[Brain Sentence {idx}] {text}")
    print(
        f"  emotion={emotion} intensity={intensity:.2f} device={DEVICE} "
        f"mouth_engine={MOUTH_ENGINE}"
    )
    print(f"{'=' * 50}")

    base.current_emotion = emotion
    base.current_intensity = intensity

    # 1) Parler Mini → WAV
    wav_path = str(TEMP_DIR / f"brain_sentence_{idx}.wav")
    style = build_voice_style(emotion, intensity)
    print(f"  [TTS/Parler] style: {style[:100]}...")
    audio, sr = generate_speech(
        text=text,
        voice_style=style,
        output_path=wav_path,
        play_audio=False,
    )
    duration = len(audio) / float(sr)
    print(f"  [TTS] Saved {wav_path} ({duration:.2f}s @ {sr} Hz)")

    # 2) emotion_26d from NVIDIA Audio2Emotion-v2.2 on this WAV
    #    (falls back to emotion_stats averages only if A2E model missing)
    print("  [emotion_26d] Audio2Emotion-v2.2 from audio (not label averages)...")
    e26 = brain_inference.get_emotion_vector(
        emotion_label=emotion,
        intensity=intensity,
        device=str(DEVICE),
        wav_path=wav_path,
    )
    print(
        f"  [emotion_26d] shape={tuple(e26.shape)} L2={e26.norm():.4f} "
        f"mean={e26.mean():.4f} explicit[16:26]={e26[16:].detach().cpu().numpy().round(3).tolist()}"
    )
    if float(e26.abs().sum()) < 1e-8:
        raise RuntimeError(
            "emotion_26d is all zeros — install Audio2Emotion "
            "(python audio2emotion.py --download) or check emotion_stats.json fallback"
        )

    brain_frames: list[dict] = []
    brain_raw = None
    w2a_frames: list[dict] = []
    fps = 30.0

    need_brain = MOUTH_ENGINE in ("hybrid", "brain")
    need_w2a = MOUTH_ENGINE in ("hybrid", "wav2arkit")

    # 3a) Brain (upper / full face) — uses same A2E 26d inside run_brain
    if need_brain:
        print("  [Brain] run_brain (HuBERT + prosody + A2E 26d + weights)...")
        brain_frames, fps, brain_raw = brain_inference.run_brain(
            wav_path=wav_path,
            emotion_label=emotion,
            intensity=intensity,
            device=str(DEVICE),
            mouth_only=False,
        )
        print(
            f"  [Brain] frames={len(brain_frames)} fps={fps} raw={brain_raw.shape} "
            f"jawOpen_peak={brain_raw[:, 24].max():.3f} "
            f"jawOpen_std={brain_raw[:, 24].std():.4f} "
            f"browInnerUp_peak={brain_raw[:, 2].max():.3f}"
        )

    # 3b) wav2arkit mouth — continuous audio→ARKit lip-sync (what looked good before)
    if need_w2a:
        print("  [LipSync] wav2arkit mouth (enhanced)...")
        w2a_frames, w2a_fps, w2a_raw = wav2arkit.audio_file_to_frames(
            wav_path,
            mouth_only=True,
            enhance_mouth=True,
        )
        fps = float(w2a_fps) or fps
        print(
            f"  [LipSync] wav2arkit frames={len(w2a_frames)} "
            f"jawOpen_peak={w2a_raw[:, 24].max():.3f} "
            f"jawOpen_std={w2a_raw[:, 24].std():.4f}"
        )

    # 3c) Merge by mode
    if MOUTH_ENGINE == "hybrid":
        frames = _merge_mouth_upper(w2a_frames, brain_frames, fps=fps)
        print(
            f"  [Merge] hybrid: mouth=wav2arkit ({len(w2a_frames)} fr) "
            f"+ upper=Brain ({len(brain_frames)} fr) → {len(frames)} fr"
        )
    elif MOUTH_ENGINE == "wav2arkit":
        frames = list(w2a_frames)
        print(f"  [Merge] wav2arkit-only mouth frames={len(frames)}")
    else:
        frames = list(brain_frames)
        print(
            f"  [Merge] pure Brain mouth (low dynamics — set MOUTH_ENGINE=hybrid for good sync)"
        )

    # 4) Play + dual UDP
    play_brain_output(
        wav_path,
        frames,
        device=DEVICE,
        emotion=emotion,
        intensity=intensity,
        fps=fps,
    )


def startup() -> None:
    print("=" * 60)
    print("ORCHESTRATOR_BRAIN — Parler Mini + Brain upper + lip-sync")
    print(f"  MOUTH_ENGINE={MOUTH_ENGINE}  "
          f"(hybrid=wav2arkit lips+Brain face | brain | wav2arkit)")
    print("=" * 60)

    if DEVICE.type == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"Using device: {DEVICE} ({name})")
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"  VRAM free/total: {free/1e9:.2f} / {total/1e9:.2f} GB")
        except Exception:
            pass
    else:
        print("WARNING: No GPU found, using CPU")
        print("Expect slower inference (HuBERT + crepe + Parler)")

    print(">>> Blender first: blender_receiver.py → bpy.ops.face.stream_receiver()")
    print("=" * 60)

    # Audio2Emotion-v2.2 (live emotion_26d from audio — primary)
    try:
        import audio2emotion as a2e

        if not a2e.is_available():
            print("Audio2Emotion model not on disk yet — attempting download (~1.27 GB)...")
            print("  Need: huggingface-cli login + accept license on")
            print("  https://huggingface.co/nvidia/Audio2Emotion-v2.2")
            try:
                a2e.ensure_model()
            except Exception as e:
                print(f"  [WARN] A2E download failed: {e}")
                print("  Falling back to emotion_stats.json averages until model is installed.")
        if a2e.is_available():
            a2e.load_session()
            print("Audio2Emotion-v2.2 ready → emotion_26d from audio")
        else:
            print("Audio2Emotion NOT ready — using emotion_stats averages")
            emo_mgr = get_emotion_manager()
            print(f"  stats labels: {sorted(emo_mgr.stats.keys())}")
    except Exception as e:
        print(f"[WARN] audio2emotion import/load: {e}")
        emo_mgr = get_emotion_manager()
        print(f"  Fallback stats labels: {sorted(emo_mgr.stats.keys())}")

    # Brain three modules (upper face / full model)
    if MOUTH_ENGINE in ("hybrid", "brain"):
        brain_inference.load_brain_model(str(DEVICE))
        print("Brain model loaded (shared_encoder + face_head + character_adapter)")
        if DEVICE.type == "cuda":
            try:
                used = torch.cuda.memory_allocated() / 1e9
                print(f"  VRAM used after Brain load: {used:.2f} GB")
            except Exception:
                pass

    # wav2arkit mouth (ONNX CPU) — the continuous lip-sync you liked
    if MOUTH_ENGINE in ("hybrid", "wav2arkit"):
        wav2arkit.load_session()
        print("wav2arkit lip-sync ready (mouth engine)")

    # Parler Mini
    load_parler()
    print("Parler-TTS Mini loaded")
    if DEVICE.type == "cuda":
        try:
            used = torch.cuda.memory_allocated() / 1e9
            print(f"  VRAM used after Parler load: {used:.2f} GB")
        except Exception:
            pass

    send_rest_pose(smooth=False)
    print("Ready. Paste JSON (blank line to submit). Type quit to exit.\n")
    if MOUTH_ENGINE == "hybrid":
        print("  Tip: lips = wav2arkit (good sync), brows/eyes = Brain + emotion_map")
        print("  Pure Brain mouth: set env MOUTH_ENGINE=brain (weaker lip-sync)\n")


def main() -> None:
    startup()
    input("Press ENTER after Blender receiver is running... ")

    # Hook process path used by base main loop? We run our own loop for full control.
    sentence_counter = 0
    while True:
        sentences = base.get_sentences_from_raw_json()
        if sentences is None:
            print("Goodbye!")
            break
        for sent in sentences:
            sentence_counter += 1
            process_sentence_brain(sent, sentence_counter)
        time.sleep(0.2)

    sd.stop()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()
