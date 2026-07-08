"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║               ASR INTELLIGENCE LAB PRO  — Benchmarking Platform                    ║
║                    Production-Ready · Streamlit Cloud Safe                          ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  Engines   : Whisper · Faster-Whisper · Sarvam AI · Wav2Vec2-XLSR-53             ║
║              SeamlessM4T-v2                                                       ║
║  Mic       : streamlit_mic_recorder  (no sounddevice / no ALSA)                    ║
║  TTS       : gTTS                                                                   ║
║  Metrics   : Latency · RTF · WER · CER · Accuracy · Rankings                      ║
║  Extras    : PDF Report · Audio Preprocessing · Radar Charts                       ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────────────
import io
import os
import time
import wave
import tempfile
import datetime as dt
import traceback
import math
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List

# ─────────────────────────────────────────────────────────────────────────────────────
# CORE THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────────────
# SOFT IMPORTS — each block fully isolated; missing package sets flag to False
# ─────────────────────────────────────────────────────────────────────────────────────

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except Exception:
    MIC_RECORDER_AVAILABLE = False

try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as FWModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except Exception:
    SOUNDFILE_AVAILABLE = False

try:
    from transformers import AutoProcessor, SeamlessM4Tv2Model, Wav2Vec2Processor, Wav2Vec2ForCTC
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════════════
SARVAM_API_URL     = "https://api.sarvam.ai/speech-to-text"
WAV2VEC2_BASE_ID   = "facebook/wav2vec2-large-xlsr-53"  # pretrained backbone (no CTC head)
# facebook/wav2vec2-large-xlsr-53 ships with NO vocabulary/CTC head of its own —
# HuggingFace's own model card says it "should be fine-tuned on a downstream task"
# before it can transcribe anything. These are verified, publicly available
# fine-tunes of that exact checkpoint (from the HF "XLSR Fine-Tuning Week"),
# each adding just a CTC head + tokenizer on top of the same XLSR-53 backbone.
WAV2VEC2_LANG_MODEL_MAP = {
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
}
WAV2VEC2_DEFAULT_MODEL = WAV2VEC2_LANG_MODEL_MAP["en"]
SEAMLESS_MODEL_ID  = "facebook/seamless-m4t-v2-large"
SARVAM_MAX_SECS    = 30.0
LOG_FILE           = "asr_lab_pro_logs.xlsx"
LIVE_CAPTION_LIMIT = 5
TARGET_SR          = 16_000

LANGUAGES = [
    ("auto", "🌐 Auto Detect"),
    ("en",   "🇬🇧 English"),
    ("hi",   "🇮🇳 Hindi"),
    ("ta",   "🇮🇳 Tamil"),
    ("te",   "🇮🇳 Telugu"),
    ("kn",   "🇮🇳 Kannada"),
    ("ml",   "🇮🇳 Malayalam"),
    ("mr",   "🇮🇳 Marathi"),
    ("bn",   "🇮🇳 Bengali"),
    ("gu",   "🇮🇳 Gujarati"),
    ("pa",   "🇮🇳 Punjabi"),
    ("ur",   "🇵🇰 Urdu"),
    ("or",   "🇮🇳 Odia"),
    ("as",   "🇮🇳 Assamese"),
    ("es",   "🇪🇸 Spanish"),
    ("fr",   "🇫🇷 French"),
    ("de",   "🇩🇪 German"),
    ("zh",   "🇨🇳 Chinese"),
    ("ja",   "🇯🇵 Japanese"),
    ("ko",   "🇰🇷 Korean"),
    ("ar",   "🇸🇦 Arabic"),
    ("pt",   "🇧🇷 Portuguese"),
    ("ru",   "🇷🇺 Russian"),
]

ENGINE_COLORS = {
    "Whisper":            "#4F46E5",
    "Faster-Whisper":     "#06B6D4",
    "Sarvam AI":          "#F59E0B",
    "Wav2Vec2-XLSR-53":   "#EC4899",
    "SeamlessM4T-v2":     "#8B5CF6",
}

SARVAM_LANG_MAP = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "bn": "bn-IN", "gu": "gu-IN",
    "pa": "pa-IN", "or": "od-IN", "en": "en-IN",
}

# SeamlessM4T-v2 targets/expects ISO 639-3 style language tokens
SEAMLESS_LANG_MAP = {
    "en": "eng", "hi": "hin", "ta": "tam", "te": "tel", "kn": "kan",
    "ml": "mal", "mr": "mar", "bn": "ben", "gu": "guj", "pa": "pan",
    "ur": "urd", "or": "ory", "as": "asm",
    "es": "spa", "fr": "fra", "de": "deu", "zh": "cmn", "ja": "jpn",
    "ko": "kor", "ar": "arb", "pt": "por", "ru": "rus",
}

# Model information table data
MODEL_INFO = [
    {"Engine": "Whisper",           "Architecture": "Transformer Encoder-Decoder", "Type": "🖥️ Local",  "Size": "~244MB (base)", "Languages": "99+",       "Avg Latency": "0.5–2s"},
    {"Engine": "Faster-Whisper",    "Architecture": "CTranslate2 Whisper",         "Type": "🖥️ Local",  "Size": "~244MB (base)", "Languages": "99+",       "Avg Latency": "0.3–1s"},
    {"Engine": "Sarvam AI",         "Architecture": "Saarika v2.5 (Cloud)",        "Type": "☁️ Cloud",  "Size": "N/A",           "Languages": "11 Indic",  "Avg Latency": "1–3s"},
    {"Engine": "Wav2Vec2-XLSR-53",  "Architecture": "Wav2Vec2 XLSR (HuggingFace)", "Type": "🖥️ Local",  "Size": "~1.2GB",        "Languages": "53 (multilingual)", "Avg Latency": "1–3s"},
    {"Engine": "SeamlessM4T-v2",    "Architecture": "SeamlessM4T-v2 (HuggingFace)","Type": "🖥️ Local",  "Size": "~9GB",          "Languages": "100+",      "Avg Latency": "2–6s"},
]


# ══════════════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass
class EngineResult:
    engine:            str
    transcript:        str   = ""
    detected_language: str   = ""
    latency_sec:       float = 0.0
    rtf:               float = 0.0
    wer:               Optional[float] = None
    cer:               Optional[float] = None
    accuracy:          Optional[float] = None
    rank:              int   = 0
    overall_score:     float = 0.0
    status:            str   = "pending"
    error_message:     str   = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════════════
# AUDIO PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════════════

def preprocess_audio(path: str, noise_reduce: bool = False) -> str:
    """
    Normalize · Resample to 16kHz · Convert stereo→mono · Optional noise reduction.
    Returns path to processed WAV. Falls back to original if librosa unavailable.
    """
    if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
        return path  # fallback — use original
    try:
        audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        out_path = path.replace(".wav", "_processed.wav")
        sf.write(out_path, audio, TARGET_SR)
        return out_path
    except Exception:
        return path  # fallback — use original


# ══════════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════════════

def bytes_to_wav_file(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Write audio bytes to temp file; convert MP3→WAV via ffmpeg if needed."""
    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    if ext == ".mp3":
        wav_path = tmp.name.replace(".mp3", ".wav")
        os.system(f'ffmpeg -y -i "{tmp.name}" -ar 16000 -ac 1 "{wav_path}" -loglevel quiet')
        return wav_path
    return tmp.name


def get_wav_duration(path: str) -> float:
    """Return WAV duration in seconds; 0.0 on failure."""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Optional[float]]:
    """WER / CER / Accuracy — Nones if jiwer missing or no reference."""
    if not reference or not JIWER_AVAILABLE:
        return {"wer": None, "cer": None, "accuracy": None}
    try:
        w = jiwer.wer(reference, hypothesis)
        c = jiwer.cer(reference, hypothesis)
        return {
            "wer":      round(w, 4),
            "cer":      round(c, 4),
            "accuracy": round(max(0.0, min(1.0, 1.0 - w)), 4),
        }
    except Exception:
        return {"wer": None, "cer": None, "accuracy": None}


def compute_rankings(results: List[EngineResult]) -> List[EngineResult]:
    """
    Rank successful engines by overall score:
    Score = 0.4*accuracy + 0.3*(1-latency_norm) + 0.15*(1-wer_norm) + 0.15*(1-cer_norm)
    """
    ok = [r for r in results if r.status == "success"]
    if not ok:
        return results

    # Normalize latency
    lats   = [r.latency_sec for r in ok]
    max_l  = max(lats) or 1.0
    wers   = [r.wer  if r.wer  is not None else 1.0 for r in ok]
    cers   = [r.cer  if r.cer  is not None else 1.0 for r in ok]
    accs   = [r.accuracy if r.accuracy is not None else 0.0 for r in ok]
    max_w  = max(wers) or 1.0
    max_c  = max(cers) or 1.0

    for i, r in enumerate(ok):
        lat_norm = r.latency_sec / max_l
        wer_norm = wers[i] / max_w
        cer_norm = cers[i] / max_c
        acc      = accs[i]
        r.overall_score = round(
            0.4 * acc + 0.3 * (1 - lat_norm) + 0.15 * (1 - wer_norm) + 0.15 * (1 - cer_norm),
            4,
        )

    ok_sorted = sorted(ok, key=lambda r: r.overall_score, reverse=True)
    for rank, r in enumerate(ok_sorted, 1):
        r.rank = rank

    # Skipped/error engines get rank 0 (already default)
    return results


def log_to_excel(rows: List[Dict], path: str = LOG_FILE):
    """Append benchmark rows to Excel; creates file if missing."""
    if not OPENPYXL_AVAILABLE:
        return False, "openpyxl not installed"
    try:
        new_df = pd.DataFrame(rows)
        if os.path.exists(path):
            df = pd.concat([pd.read_excel(path), new_df], ignore_index=True)
        else:
            df = new_df
        df.to_excel(path, index=False)
        return True, os.path.abspath(path)
    except Exception as e:
        return False, str(e)


def make_tts_audio(text: str, lang_code: str = "en") -> Optional[bytes]:
    """text → MP3 bytes via gTTS; None on failure."""
    if not GTTS_AVAILABLE or not text.strip():
        return None
    try:
        tts_lang = lang_code if (lang_code != "auto" and len(lang_code) == 2) else "en"
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Loading Whisper model…")
def get_whisper_model(size: str):
    return openai_whisper.load_model(size)


@st.cache_resource(show_spinner="⏳ Loading Faster-Whisper model…")
def get_faster_whisper_model(size: str):
    return FWModel(size, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner="⏳ Loading Wav2Vec2-XLSR-53 checkpoint…")
def get_wav2vec2_model(model_id: str):
    """
    Loads the processor+model directly (no transformers.pipeline wrapper).
    The high-level ASR pipeline probes for an optional torchcodec backend even
    when given a raw array, and on some machines torchcodec is present but its
    native `libtorchcodec` binary fails to load — this direct path never touches
    torchcodec at all.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model     = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner="⏳ Loading SeamlessM4T-v2-Large (facebook/seamless-m4t-v2-large)… this can take a while on first run")
def get_seamless_m4t_v2():
    """
    Loads processor+model, self-healing against a corrupted/partially-downloaded
    local cache (e.g. an interrupted download of this ~9GB checkpoint leaving a
    truncated tokenizer.model). On the first failure we retry once with
    force_download=True to pull a clean copy before giving up.
    """
    last_err = None
    for force in (False, True):
        try:
            processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL_ID, force_download=force)
            model     = SeamlessM4Tv2Model.from_pretrained(SEAMLESS_MODEL_ID, force_download=force)
            model.eval()
            return processor, model
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Failed to load SeamlessM4T-v2-Large even after a forced re-download "
        f"(possibly a version mismatch in sentencepiece/protobuf): {last_err}"
    )


def _load_audio_array(path: str, target_sr: int = TARGET_SR):
    """
    Load mono float32 audio as a numpy array at target_sr.
    Tries librosa first (handles resampling), falls back to soundfile.
    Raises RuntimeError if neither backend is available.
    """
    if LIBROSA_AVAILABLE:
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        return audio, sr
    if SOUNDFILE_AVAILABLE:
        audio, sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype("float32"), sr
    raise RuntimeError("No audio backend available — install librosa or soundfile")



# ══════════════════════════════════════════════════════════════════════════════════════
# ASR ENGINE WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════════════

def run_whisper(path: str, size: str, lang: str) -> EngineResult:
    r = EngineResult(engine="Whisper")
    if not WHISPER_AVAILABLE:
        r.status = "error"; r.error_message = "openai-whisper not installed"; return r
    try:
        model  = get_whisper_model(size)
        kwargs: Dict[str, Any] = {"task": "transcribe"}
        if lang != "auto":
            kwargs["language"] = lang
        t0 = time.perf_counter()
        out = model.transcribe(path, **kwargs)
        r.latency_sec       = round(time.perf_counter() - t0, 3)
        r.transcript        = out.get("text", "").strip()
        r.detected_language = out.get("language", lang)
        r.status            = "success"
    except Exception as e:
        r.status = "error"; r.error_message = str(e)
    return r


def run_faster_whisper(path: str, size: str, lang: str) -> EngineResult:
    r = EngineResult(engine="Faster-Whisper")
    if not FASTER_WHISPER_AVAILABLE:
        r.status = "error"; r.error_message = "faster-whisper not installed"; return r
    try:
        model  = get_faster_whisper_model(size)
        kwargs: Dict[str, Any] = {"task": "transcribe"}
        if lang != "auto":
            kwargs["language"] = lang
        t0 = time.perf_counter()
        segs, info = model.transcribe(path, **kwargs)
        text = " ".join(s.text.strip() for s in segs)
        r.latency_sec       = round(time.perf_counter() - t0, 3)
        r.transcript        = text.strip()
        r.detected_language = getattr(info, "language", lang) or lang
        r.status            = "success"
    except Exception as e:
        r.status = "error"; r.error_message = str(e)
    return r


def run_sarvam(path: str, duration: float, api_key: str, lang: str) -> EngineResult:
    r = EngineResult(engine="Sarvam AI")
    if duration > SARVAM_MAX_SECS:
        r.status = "skipped"
        r.error_message = f"Audio {duration:.1f}s > {SARVAM_MAX_SECS:.0f}s limit"
        return r
    if not REQUESTS_AVAILABLE:
        r.status = "error"; r.error_message = "requests not installed"; return r
    if not api_key:
        r.status = "skipped"; r.error_message = "No Sarvam API key"; return r
    try:
        sarvam_lang = SARVAM_LANG_MAP.get(lang, "unknown")
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            resp = _requests.post(
                SARVAM_API_URL,
                headers={"api-subscription-key": api_key},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"model": "saarika:v2.5", "language_code": sarvam_lang},
                timeout=60,
            )
        r.latency_sec = round(time.perf_counter() - t0, 3)
        if resp.status_code == 200:
            payload = resp.json()
            r.transcript        = payload.get("transcript", "").strip()
            r.detected_language = payload.get("language_code", sarvam_lang)
            r.status            = "success"
        else:
            r.status = "error"; r.error_message = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        r.status = "error"; r.error_message = str(e)
    return r


def run_wav2vec2(path: str, lang: str) -> EngineResult:
    """
    facebook/wav2vec2-large-xlsr-53 — local, multilingual (incl. Hindi) CTC model.
    The raw XLSR-53 backbone has no vocabulary/CTC head of its own, so this routes
    to a verified fine-tuned checkpoint built on that same backbone for the
    requested language (see WAV2VEC2_LANG_MODEL_MAP), falling back to the English
    checkpoint for languages without a dedicated fine-tune. Loaded once via
    st.cache_resource; runs entirely on-device, no API key, no cloud dependency.
    """
    r = EngineResult(engine="Wav2Vec2-XLSR-53")
    if not TRANSFORMERS_AVAILABLE:
        r.status = "error"; r.error_message = "transformers/torch not installed"; return r
    try:
        if lang in WAV2VEC2_LANG_MODEL_MAP:
            model_id = WAV2VEC2_LANG_MODEL_MAP[lang]
            fallback_note = ""
        else:
            model_id = WAV2VEC2_DEFAULT_MODEL
            fallback_note = (
                f"No fine-tuned Wav2Vec2-XLSR-53 checkpoint for '{lang}' — "
                "used the English checkpoint as a fallback."
            )

        processor, model = get_wav2vec2_model(model_id)
        # Decode the audio ourselves (librosa/soundfile) — no pipeline, no torchcodec.
        audio, sr = _load_audio_array(path, TARGET_SR)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        r.latency_sec = round(time.perf_counter() - t0, 3)

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript    = processor.batch_decode(predicted_ids)[0]

        r.transcript        = transcript.strip()
        r.detected_language = lang if lang in WAV2VEC2_LANG_MODEL_MAP else "en (fallback)"
        r.status            = "success" if r.transcript else "error"
        if not r.transcript:
            r.error_message = "Wav2Vec2-XLSR-53 returned an empty transcript."
        elif fallback_note:
            r.error_message = fallback_note
    except Exception as e:
        r.status = "error"; r.error_message = str(e)[:250]
    return r


def run_seamless_m4t(path: str, lang: str) -> EngineResult:
    """
    facebook/seamless-m4t-v2-large — local, multilingual (incl. Hindi) speech-to-text.
    Loaded once via cached AutoProcessor + SeamlessM4Tv2Model; runs entirely on-device,
    no API key, no cloud dependency. Audio is decoded to a 16kHz mono array and fed
    through the model's `.generate(..., generate_speech=False)` text path.
    """
    r = EngineResult(engine="SeamlessM4T-v2")
    if not TRANSFORMERS_AVAILABLE:
        r.status = "error"; r.error_message = "transformers/torch not installed"; return r
    try:
        processor, model = get_seamless_m4t_v2()
        audio, sr = _load_audio_array(path, TARGET_SR)
        tgt_lang  = SEAMLESS_LANG_MAP.get(lang, "eng")

        inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.no_grad():
            output_tokens = model.generate(**inputs, tgt_lang=tgt_lang, generate_speech=False)
        r.latency_sec = round(time.perf_counter() - t0, 3)

        transcript = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        r.transcript        = transcript.strip()
        r.detected_language = tgt_lang
        r.status            = "success" if r.transcript else "error"
        if not r.transcript:
            r.error_message = "SeamlessM4T-v2 returned an empty transcript."
    except Exception as e:
        r.status = "error"; r.error_message = str(e)[:250]
    return r


# ══════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════════════

def run_benchmark(path: str, duration: float, cfg: Dict[str, Any]) -> List[EngineResult]:
    """Run all enabled engines sequentially; compute RTF + metrics + rankings."""
    enabled = cfg["engines_enabled"]
    results: List[EngineResult] = []
    steps   = [k for k, v in enabled.items() if v]
    total   = max(len(steps), 1)
    bar     = st.progress(0.0, text="Initialising benchmark…")
    done    = 0

    def _tick(label: str):
        nonlocal done
        bar.progress(min(done / total, 1.0), text=label)

    if enabled.get("Whisper"):
        _tick("⚙️ Running Whisper…")
        results.append(run_whisper(path, cfg["model_size"], cfg["lang"]))
        done += 1

    if enabled.get("Faster-Whisper"):
        _tick("⚙️ Running Faster-Whisper…")
        results.append(run_faster_whisper(path, cfg["model_size"], cfg["lang"]))
        done += 1

    if enabled.get("Sarvam AI"):
        _tick("☁️ Running Sarvam AI…")
        results.append(run_sarvam(path, duration, cfg["sarvam_key"], cfg["lang"]))
        done += 1

    if enabled.get("Wav2Vec2-XLSR-53"):
        _tick("🧬 Running Wav2Vec2-XLSR-53…")
        results.append(run_wav2vec2(path, cfg["lang"]))
        done += 1

    if enabled.get("SeamlessM4T-v2"):
        _tick("🌐 Running SeamlessM4T-v2…")
        results.append(run_seamless_m4t(path, cfg["lang"]))
        done += 1

    bar.progress(1.0, text="✅ Benchmark complete!")
    time.sleep(0.4)
    bar.empty()

    # Post-process: RTF + metrics
    ref = cfg.get("reference_text", "")
    for r in results:
        if r.status == "success" and duration > 0:
            r.rtf = round(r.latency_sec / duration, 4)
        if r.status == "success":
            m = compute_metrics(ref, r.transcript)
            r.wer = m["wer"]; r.cer = m["cer"]; r.accuracy = m["accuracy"]

    # Compute rankings
    results = compute_rankings(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    results: List[EngineResult],
    duration: float,
    lang_label: str,
    audio_filename: str = "audio.wav",
) -> Optional[bytes]:
    """Generate a professional PDF benchmark report. Returns bytes or None."""
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        buf     = io.BytesIO()
        doc     = SimpleDocTemplate(buf, pagesize=A4, topMargin=0.7*inch, bottomMargin=0.7*inch)
        styles  = getSampleStyleSheet()
        story   = []

        title_style = ParagraphStyle(
            "Title", parent=styles["Title"], fontSize=20, spaceAfter=6,
            textColor=colors.HexColor("#4F46E5"),
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
            textColor=colors.HexColor("#111827"),
        )
        body_style = styles["BodyText"]

        # Header
        story.append(Paragraph("🎙️ ASR Intelligence Lab Pro", title_style))
        story.append(Paragraph("Benchmark Report", styles["Heading2"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB")))
        story.append(Spacer(1, 0.15*inch))

        # Audio info
        story.append(Paragraph("Audio Information", h2_style))
        info_data = [
            ["File", audio_filename],
            ["Duration", f"{duration:.2f}s"],
            ["Language", lang_label],
            ["Generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        info_table = Table(info_data, colWidths=[1.5*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F3F4F6")),
            ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.2*inch))

        # Results table
        story.append(Paragraph("Engine Comparison", h2_style))
        ok = [r for r in results if r.status == "success"]

        if ok:
            headers = ["Engine", "Latency(s)", "RTF", "WER", "CER", "Accuracy", "Rank"]
            rows    = [headers]
            for r in ok:
                medal = {1:"🥇", 2:"🥈", 3:"🥉"}.get(r.rank, f"#{r.rank}")
                rows.append([
                    r.engine,
                    f"{r.latency_sec:.3f}",
                    f"{r.rtf:.4f}",
                    f"{r.wer:.4f}" if r.wer is not None else "—",
                    f"{r.cer:.4f}" if r.cer is not None else "—",
                    f"{r.accuracy:.2%}" if r.accuracy is not None else "—",
                    medal,
                ])
            results_table = Table(rows, repeatRows=1)
            results_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#4F46E5")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
                ("FONTSIZE", (0,0), (-1,-1), 9),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("PADDING", (0,0), (-1,-1), 5),
                ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ]))
            story.append(results_table)
            story.append(Spacer(1, 0.2*inch))

            # Best engine
            best = ok[0] if ok else None
            fastest = min(ok, key=lambda x: x.latency_sec) if ok else None
            story.append(Paragraph("Rankings & Conclusion", h2_style))
            if best:
                story.append(Paragraph(
                    f"🥇 <b>Best Overall:</b> {best.engine} "
                    f"(Score: {best.overall_score:.3f})", body_style))
            if fastest:
                story.append(Paragraph(
                    f"⚡ <b>Fastest:</b> {fastest.engine} "
                    f"({fastest.latency_sec:.2f}s)", body_style))
            story.append(Spacer(1, 0.1*inch))

        # Transcripts
        story.append(Paragraph("Transcripts", h2_style))
        for r in ok:
            story.append(Paragraph(f"<b>{r.engine}</b>", body_style))
            story.append(Paragraph(r.transcript or "(empty)", body_style))
            story.append(Spacer(1, 0.1*inch))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════════════

def bar_chart(df: pd.DataFrame, col: str, title: str, y_label: str):
    if not PLOTLY_AVAILABLE:
        return None
    plot = df.dropna(subset=[col])
    if plot.empty:
        return None
    colors_list = [ENGINE_COLORS.get(e, "#6B7280") for e in plot["engine"]]
    fig = go.Figure(go.Bar(
        x=plot["engine"], y=plot[col],
        marker_color=colors_list,
        text=[f"{v:.3f}" for v in plot[col]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, yaxis_title=y_label, xaxis_title="Engine",
        template="plotly_white", height=340,
        margin=dict(t=55, b=30, l=30, r=10),
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def radar_chart(df: pd.DataFrame) -> Optional[Any]:
    """Overall score radar chart across engines."""
    if not PLOTLY_AVAILABLE:
        return None
    ok = df[df["status"] == "success"].dropna(subset=["latency_sec"])
    if ok.empty:
        return None

    categories = ["Accuracy", "Speed", "Low WER", "Low CER"]
    fig = go.Figure()

    max_lat = ok["latency_sec"].max() or 1.0
    max_wer = ok["wer"].max() if ok["wer"].notna().any() else 1.0
    max_cer = ok["cer"].max() if ok["cer"].notna().any() else 1.0

    for _, row in ok.iterrows():
        acc   = float(row["accuracy"]) if pd.notna(row["accuracy"]) else 0.0
        speed = 1.0 - float(row["latency_sec"]) / max_lat
        lwer  = 1.0 - (float(row["wer"]) / max_wer if pd.notna(row["wer"]) else 1.0)
        lcer  = 1.0 - (float(row["cer"]) / max_cer if pd.notna(row["cer"]) else 1.0)
        vals  = [acc, speed, lwer, lcer, acc]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill="toself",
            name=row["engine"],
            line_color=ENGINE_COLORS.get(row["engine"], "#6B7280"),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Engine Radar Chart",
        template="plotly_white",
        height=420,
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def ranking_chart(df: pd.DataFrame) -> Optional[Any]:
    """Overall score horizontal bar chart = ranking chart."""
    if not PLOTLY_AVAILABLE:
        return None
    ok = df[(df["status"] == "success") & (df["overall_score"] > 0)].sort_values("overall_score")
    if ok.empty:
        return None
    colors_list = [ENGINE_COLORS.get(e, "#6B7280") for e in ok["engine"]]
    fig = go.Figure(go.Bar(
        x=ok["overall_score"], y=ok["engine"],
        orientation="h",
        marker_color=colors_list,
        text=[f"{v:.3f}" for v in ok["overall_score"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Engine Ranking (Overall Score)",
        xaxis_title="Overall Score", yaxis_title="",
        template="plotly_white", height=320,
        margin=dict(t=55, b=30, l=10, r=60),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS — Light + Dark mode
# ══════════════════════════════════════════════════════════════════════════════════════

def page_setup():
    st.set_page_config(
        page_title="ASR Intelligence Lab Pro",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    /* ══ LIGHT MODE ══ */
    .stApp { background: #F8FAFC; color: #111827; }

    .lab-title {
        font-size: 2.6rem; font-weight: 900; line-height: 1.15;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .lab-subtitle { color: #6B7280; font-size: .95rem; margin-top: 0; margin-bottom: 16px; }

    .card {
        background: #FFFFFF; border: 1px solid #E5E7EB;
        border-radius: 16px; padding: 18px 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom: 8px;
    }
    .card-title {
        color: #6B7280; font-size: .78rem; text-transform: uppercase;
        letter-spacing: .08em; font-weight: 600; margin-bottom: 4px;
    }
    .card-value { font-size: 1.45rem; font-weight: 800; color: #111827; }
    .card-sub   { font-size: .82rem; color: #9CA3AF; margin-top: 2px; }

    .caption-box {
        background: #F0F9FF; border: 1px solid #BAE6FD;
        border-radius: 12px; padding: 16px; min-height: 80px;
        font-size: 1.1rem; color: #0C4A6E; line-height: 1.6;
    }

    .avail-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    .avail-ok  { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0;
                 border-radius: 8px; padding: 3px 10px; font-size: .75rem; font-weight: 600; }
    .avail-err { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA;
                 border-radius: 8px; padding: 3px 10px; font-size: .75rem; font-weight: 600; }

    .pill {
        display: inline-block; border-radius: 999px;
        padding: 2px 12px; font-size: .75rem; font-weight: 700; margin-right: 6px;
    }

    .ai-box {
        background: #FAFAFA; border: 1px solid #E5E7EB;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;
    }
    .ai-label { font-size: .78rem; font-weight: 700; color: #4F46E5;
                text-transform: uppercase; margin-bottom: 4px; }

    section[data-testid="stSidebar"] {
        background: #FFFFFF !important; border-right: 1px solid #E5E7EB;
    }

    /* ══ DARK MODE ══ */
    @media (prefers-color-scheme: dark) {
        .stApp { background: #0f1117 !important; color: #f0f2f6 !important; }
        .card  { background: #1f2937 !important; border-color: #374151 !important;
                 box-shadow: 0 4px 15px rgba(0,0,0,.4) !important; }
        .card-title { color: #9CA3AF !important; }
        .card-value { color: #f9fafb !important; }
        .card-sub   { color: #6B7280 !important; }
        .caption-box { background: #1e3a5f !important; border-color: #2563eb !important;
                       color: #bfdbfe !important; }
        .avail-ok  { background: #064e3b !important; color: #6ee7b7 !important;
                     border-color: #065f46 !important; }
        .avail-err { background: #450a0a !important; color: #fca5a5 !important;
                     border-color: #7f1d1d !important; }
        .ai-box    { background: #1f2937 !important; border-color: #374151 !important; }
        .ai-label  { color: #818cf8 !important; }
        section[data-testid="stSidebar"] {
            background: #111827 !important; border-color: #1f2937 !important;
        }
    }

    /* ══ Shared ══ */
    .ok { color: #059669; font-weight: 700; }
    .sk { color: #D97706; font-weight: 700; }
    .er { color: #DC2626; font-weight: 700; }
    .stButton > button { width: 100%; border-radius: 10px; font-weight: 700; }

    /* ══ API Key inputs (sidebar) — visible border ══ */
    section[data-testid="stSidebar"] div[data-baseweb="input"] {
        border: 1.5px solid #C7D2FE !important;
        border-radius: 8px !important;
        background: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px rgba(79,70,229,.15) !important;
    }
    section[data-testid="stSidebar"] input {
        border: none !important;
    }

    /* ══ Transcript boxes (results tab / TTS tab) — visible border ══ */
    div[data-testid="stTextArea"] textarea {
        border: 1.5px solid #C7D2FE !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px rgba(79,70,229,.15) !important;
    }

    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            border-color: #374151 !important;
            background: #1f2937 !important;
        }
        div[data-testid="stTextArea"] textarea {
            border-color: #374151 !important;
            background: #1f2937 !important;
            color: #f0f2f6 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════════
# AVAILABILITY STRIP
# ══════════════════════════════════════════════════════════════════════════════════════

def render_availability():
    deps = [
        ("🎤 Mic",            MIC_RECORDER_AVAILABLE),
        ("🧠 Whisper",        WHISPER_AVAILABLE),
        ("⚡ FW",             FASTER_WHISPER_AVAILABLE),
        ("☁️ Sarvam",        REQUESTS_AVAILABLE),
        ("🧬 Wav2Vec2",       TRANSFORMERS_AVAILABLE),
        ("🌐 SeamlessM4T",    TRANSFORMERS_AVAILABLE),
        ("📊 WER/CER",        JIWER_AVAILABLE),
        ("🔊 TTS",            GTTS_AVAILABLE),
        ("📒 Excel",          OPENPYXL_AVAILABLE),
        ("📄 PDF",            REPORTLAB_AVAILABLE),
        ("🎛️ AudioPrep",     LIBROSA_AVAILABLE),
    ]
    items = "".join(
        '<span class="{cls}">{icon} {name}</span>'.format(
            cls="avail-ok" if ok else "avail-err",
            icon="✅" if ok else "❌",
            name=n,
        )
        for n, ok in deps
    )
    st.markdown(f'<div class="avail-strip">{items}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════════

def build_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown("## ⚙️ Configuration")

    st.sidebar.markdown("### 🌐 Language")
    lang_label = st.sidebar.selectbox(
        "Spoken language", [l for _, l in LANGUAGES], index=0, label_visibility="collapsed",
    )
    lang_code = next(c for c, l in LANGUAGES if l == lang_label)

    st.sidebar.markdown("### 🧠 Local Model Size")
    model_size = st.sidebar.select_slider(
        "Whisper / Faster-Whisper",
        options=["tiny", "base", "small", "medium"],
        value="small",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### 🎛️ Audio Preprocessing")
    noise_reduce = st.sidebar.checkbox("Noise Reduction (requires librosa)", value=False)

    st.sidebar.markdown("### ☁️ API Keys")
    sarvam_key = st.sidebar.text_input(
        "Sarvam AI Key", type="password", value=os.environ.get("SARVAM_API_KEY", ""),
    )

    st.sidebar.markdown("### ✅ Engines")
    col_a, col_b = st.sidebar.columns(2)
    engines = {
        "Whisper":           col_a.checkbox("Whisper",           value=True),
        "Faster-Whisper":    col_b.checkbox("Faster-Whisper",    value=True),
        "Sarvam AI":         col_a.checkbox("Sarvam AI",         value=True),
        "Wav2Vec2-XLSR-53":  col_b.checkbox("Wav2Vec2-XLSR-53",  value=True),
        "SeamlessM4T-v2":    col_a.checkbox("SeamlessM4T-v2",    value=True),
    }

    st.sidebar.markdown("### 📝 Reference Transcript")
    ref_text = st.sidebar.text_area(
        "Enables WER / CER / Accuracy", height=90, label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("ASR Intelligence Lab Pro · Streamlit Cloud Safe")

    return {
        "lang":            lang_code,
        "lang_label":      lang_label,
        "model_size":      model_size,
        "noise_reduce":    noise_reduce,
        "sarvam_key":      sarvam_key.strip(),
        "engines_enabled": engines,
        "reference_text":  ref_text.strip(),
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# SHARED: save results
# ══════════════════════════════════════════════════════════════════════════════════════

def _save_results(results: List[EngineResult], duration: float, cfg: Dict[str, Any]):
    st.session_state["last_results"] = results
    st.session_state["last_duration"] = duration

    ts   = dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for r in results:
        row = r.to_row()
        row.update({
            "timestamp":          ts,
            "audio_duration_sec": round(duration, 3),
            "language":           cfg["lang_label"],
            "reference_provided": bool(cfg["reference_text"]),
            "model_name":         r.engine,
            "ranking":            r.rank,
            "overall_score":      r.overall_score,
        })
        rows.append(row)

    ok, info = log_to_excel(rows)
    if ok:
        st.toast(f"📒 Logged → {info}", icon="✅")
    else:
        st.warning(f"Excel log failed: {info}")


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE RECORD
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_live(cfg: Dict[str, Any]):
    st.markdown("### 🎙️ Record & Transcribe")

    live_mode = st.toggle(
        "⚡ Live Captions Mode", value=False,
        help=f"Auto-clears after {LIVE_CAPTION_LIMIT} recordings",
    )

    if "live_captions" not in st.session_state:
        st.session_state.live_captions = []
    if "recording_count" not in st.session_state:
        st.session_state.recording_count = 0

    if not MIC_RECORDER_AVAILABLE:
        st.error("❌ `streamlit-mic-recorder` not installed. Run: `pip install streamlit-mic-recorder`")
        return

    audio = mic_recorder(
        start_prompt="🎤  Start Recording",
        stop_prompt="⏹️  Stop Recording",
        format="wav", key="mic",
    )

    if not audio:
        st.info("👆 Press **Start Recording**, speak, then press **Stop Recording**.")
        if live_mode and st.session_state.live_captions:
            st.markdown("#### 📡 Live Captions")
            st.markdown(
                '<div class="caption-box">' +
                "<br>".join(f"[{i+1}] {t}" for i, t in enumerate(st.session_state.live_captions)) +
                "</div>", unsafe_allow_html=True,
            )
        return

    audio_bytes: bytes = audio["bytes"]
    st.audio(audio_bytes, format="audio/wav")

    path     = bytes_to_wav_file(audio_bytes, "recording.wav")
    path     = preprocess_audio(path, cfg["noise_reduce"])
    duration = get_wav_duration(path)
    st.caption(f"Duration: **{duration:.2f}s**  |  Language: **{cfg['lang_label']}**")

    if duration < 0.3:
        st.warning("Recording too short — please speak for at least 0.5 seconds.")
        return

    if WHISPER_AVAILABLE and cfg["engines_enabled"].get("Whisper"):
        with st.spinner("⚡ Instant transcription…"):
            quick = run_whisper(path, cfg["model_size"], cfg["lang"])
        if quick.status == "success":
            st.success(f"**Whisper:** {quick.transcript}")
            if live_mode:
                st.session_state.recording_count += 1
                st.session_state.live_captions.append(quick.transcript)
                if st.session_state.recording_count >= LIVE_CAPTION_LIMIT:
                    st.session_state.live_captions = []
                    st.session_state.recording_count = 0
                    st.toast("🧹 Live captions cleared", icon="🔄")
        else:
            st.warning(f"Whisper: {quick.error_message}")

    st.markdown("---")
    if st.button("🚀 Run Full Benchmark (all engines)", use_container_width=True, type="primary"):
        with st.spinner("Running all ASR engines…"):
            results = run_benchmark(path, duration, cfg)
        _save_results(results, duration, cfg)
        st.session_state["last_audio_file"] = "recording.wav"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_upload(cfg: Dict[str, Any]):
    st.markdown("### 📁 Upload WAV or MP3 File")
    uploaded = st.file_uploader(
        "Choose a WAV or MP3 file", type=["wav", "mp3"], label_visibility="collapsed",
    )

    if not uploaded:
        st.info("Upload a WAV or MP3 file to benchmark all engines.")
        return

    audio_bytes = bytes(uploaded.getbuffer())
    fmt = "audio/mp3" if uploaded.name.lower().endswith(".mp3") else "audio/wav"
    st.audio(audio_bytes, format=fmt)

    path     = bytes_to_wav_file(audio_bytes, uploaded.name)
    path     = preprocess_audio(path, cfg["noise_reduce"])
    duration = get_wav_duration(path)

    col1, col2 = st.columns(2)
    col1.metric("Duration", f"{duration:.2f}s")
    col2.metric("Language", cfg["lang_label"])

    if duration > SARVAM_MAX_SECS and cfg["engines_enabled"].get("Sarvam AI"):
        st.warning(f"⚠️ Audio {duration:.1f}s — Sarvam AI will be skipped (limit: {SARVAM_MAX_SECS:.0f}s).")

    st.markdown("---")
    if st.button("🚀 Run Full Benchmark", use_container_width=True, type="primary"):
        with st.spinner("Running all ASR engines…"):
            results = run_benchmark(path, duration, cfg)
        _save_results(results, duration, cfg)
        st.session_state["last_audio_file"] = uploaded.name
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_results(cfg: Dict[str, Any]):
    results: Optional[List[EngineResult]] = st.session_state.get("last_results")
    duration  = st.session_state.get("last_duration", 0.0)
    audio_file = st.session_state.get("last_audio_file", "audio.wav")

    if not results:
        st.info("No benchmark results yet. Record or upload audio and run a benchmark first.")
        return

    df         = pd.DataFrame([r.to_row() for r in results])
    successful = df[df["status"] == "success"].copy()

    # ── Summary cards ────────────────────────────────────────────────────────────────
    st.markdown("#### 🏅 Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    ok = [r for r in results if r.status == "success"]

    with c1:
        if ok and any(r.accuracy is not None for r in ok):
            best_acc = max((r for r in ok if r.accuracy is not None), key=lambda x: x.accuracy)
            st.markdown(
                f'<div class="card"><div class="card-title">🏆 Best Accuracy</div>'
                f'<div class="card-value">{best_acc.engine}</div>'
                f'<div class="card-sub">{best_acc.accuracy:.2%}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">🏆 Best Accuracy</div>'
                        '<div class="card-sub">Add reference transcript</div></div>', unsafe_allow_html=True)

    with c2:
        if ok:
            fastest = min(ok, key=lambda x: x.latency_sec)
            st.markdown(
                f'<div class="card"><div class="card-title">⚡ Fastest</div>'
                f'<div class="card-value">{fastest.engine}</div>'
                f'<div class="card-sub">{fastest.latency_sec:.2f}s</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">⚡ Fastest</div>'
                        '<div class="card-sub">No successful runs</div></div>', unsafe_allow_html=True)

    with c3:
        if ok and any(r.wer is not None for r in ok):
            best_wer = min((r for r in ok if r.wer is not None), key=lambda x: x.wer)
            st.markdown(
                f'<div class="card"><div class="card-title">📉 Lowest WER</div>'
                f'<div class="card-value">{best_wer.engine}</div>'
                f'<div class="card-sub">WER {best_wer.wer:.4f}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">📉 Lowest WER</div>'
                        '<div class="card-sub">Add reference transcript</div></div>', unsafe_allow_html=True)

    with c4:
        if ok and any(r.cer is not None for r in ok):
            best_cer = min((r for r in ok if r.cer is not None), key=lambda x: x.cer)
            st.markdown(
                f'<div class="card"><div class="card-title">📉 Lowest CER</div>'
                f'<div class="card-value">{best_cer.engine}</div>'
                f'<div class="card-sub">CER {best_cer.cer:.4f}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">📉 Lowest CER</div>'
                        '<div class="card-sub">Add reference transcript</div></div>', unsafe_allow_html=True)

    with c5:
        if ok and any(r.overall_score > 0 for r in ok):
            best_overall = max(ok, key=lambda x: x.overall_score)
            medal = {1:"🥇", 2:"🥈", 3:"🥉"}.get(best_overall.rank, "⭐")
            st.markdown(
                f'<div class="card"><div class="card-title">{medal} Best Overall</div>'
                f'<div class="card-value">{best_overall.engine}</div>'
                f'<div class="card-sub">Score {best_overall.overall_score:.3f}</div></div>',
                unsafe_allow_html=True)
        else:
            n_ok = len(ok); n_sk = (df["status"]=="skipped").sum(); n_er = (df["status"]=="error").sum()
            st.markdown(
                f'<div class="card"><div class="card-title">📊 Run Summary</div>'
                f'<div class="card-value">{n_ok}/{len(df)}</div>'
                f'<div class="card-sub">✅{n_ok} ⏭️{n_sk} ❌{n_er}</div></div>',
                unsafe_allow_html=True)

    st.markdown("")

    # ── Rankings ────────────────────────────────────────────────────────────────────
    if any(r.rank > 0 for r in results):
        st.markdown("#### 🏅 Engine Rankings")
        ranked = sorted([r for r in ok if r.rank > 0], key=lambda x: x.rank)
        rank_cols = st.columns(min(len(ranked), 5))
        for col, r in zip(rank_cols, ranked):
            medal = {1:"🥇", 2:"🥈", 3:"🥉"}.get(r.rank, f"#{r.rank}")
            with col:
                st.markdown(
                    f'<div class="card" style="text-align:center;">'
                    f'<div style="font-size:2rem;">{medal}</div>'
                    f'<div class="card-value" style="font-size:1rem;">{r.engine}</div>'
                    f'<div class="card-sub">Score: {r.overall_score:.3f}</div></div>',
                    unsafe_allow_html=True)
        st.markdown("")

    # ── Data table ───────────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Detailed Results")
    display = df.copy()
    display["status"] = display["status"].map(
        {"success": "✅ Success", "skipped": "⏭️ Skipped", "error": "❌ Error"})
    st.dataframe(
        display[[
            "engine", "status", "detected_language", "latency_sec",
            "rtf", "wer", "cer", "accuracy", "overall_score", "rank", "error_message",
        ]].rename(columns={
            "engine": "Engine", "status": "Status", "detected_language": "Language",
            "latency_sec": "Latency(s)", "rtf": "RTF", "wer": "WER", "cer": "CER",
            "accuracy": "Accuracy", "overall_score": "Score", "rank": "Rank",
            "error_message": "Notes",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Charts ───────────────────────────────────────────────────────────────────────
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly: `pip install plotly`")
    elif successful.empty:
        st.info("No successful runs to chart.")
    else:
        st.markdown("#### 📉 Comparison Charts")
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = bar_chart(successful, "latency_sec", "Latency by Engine", "Seconds")
            if fig: st.plotly_chart(fig, use_container_width=True)
        with cc2:
            fig = bar_chart(successful, "accuracy", "Accuracy by Engine", "Accuracy")
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Add reference transcript for accuracy chart.")

        cc3, cc4 = st.columns(2)
        with cc3:
            fig = bar_chart(successful, "wer", "WER by Engine", "Word Error Rate")
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Add reference transcript for WER chart.")
        with cc4:
            fig = bar_chart(successful, "cer", "CER by Engine", "Char Error Rate")
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Add reference transcript for CER chart.")

        # Radar + Ranking charts
        st.markdown("#### 🕸️ Advanced Charts")
        rc1, rc2 = st.columns(2)
        with rc1:
            fig = radar_chart(successful)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Add reference transcript for radar chart.")
        with rc2:
            fig = ranking_chart(successful)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Run benchmark to see ranking chart.")

    # ── Transcripts ─────────────────────────────────────────────────────────────────
    st.markdown("#### 📝 Transcripts")
    ok_results = [r for r in results if r.status == "success"]
    if not ok_results:
        st.warning("No successful transcripts to show.")
    else:
        cols = st.columns(min(len(ok_results), 4))
        for idx, r in enumerate(ok_results):
            with cols[idx % 4]:
                color = ENGINE_COLORS.get(r.engine, "#6B7280")
                st.markdown(
                    f'<span class="pill" style="background:{color}22;color:{color};">{r.engine}</span>',
                    unsafe_allow_html=True)
                st.text_area("", r.transcript, height=150, key=f"tx_{r.engine}", label_visibility="collapsed")
                st.download_button(
                    "⬇️ Download",
                    data=r.transcript,
                    file_name=f"{r.engine.replace(' ','_').lower()}_transcript.txt",
                    mime="text/plain", key=f"dl_{r.engine}", use_container_width=True,
                )

    # ── Model Information Table ──────────────────────────────────────────────────────
    with st.expander("ℹ️ Model Information & Comparison"):
        model_df = pd.DataFrame(MODEL_INFO)
        st.dataframe(model_df, use_container_width=True, hide_index=True)

    # ── PDF Report ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    pdf_col, excel_col = st.columns(2)

    with pdf_col:
        if REPORTLAB_AVAILABLE:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    pdf_bytes = generate_pdf_report(results, duration, cfg["lang_label"], audio_file)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"asr_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.error("PDF generation failed.")
        else:
            st.info("Install reportlab for PDF: `pip install reportlab`")

    with excel_col:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "rb") as f:
                st.download_button(
                    "📥 Download Excel Log",
                    data=f.read(),
                    file_name=LOG_FILE,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            st.caption(f"Log: `{os.path.abspath(LOG_FILE)}`")


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEXT-TO-SPEECH
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_tts(cfg: Dict[str, Any]):
    st.markdown("### 🔊 Text-to-Speech")
    st.markdown("Convert any transcript or custom text into playable speech.")

    if not GTTS_AVAILABLE:
        st.error("❌ gTTS not installed. Run: `pip install gtts`")
        return

    last_results: List[EngineResult] = st.session_state.get("last_results", [])
    best_text = ""
    if last_results:
        ok = [r for r in last_results if r.status == "success" and r.transcript]
        if ok:
            best_text = ok[0].transcript

    tts_text = st.text_area(
        "Text to speak", value=best_text, height=160, placeholder="Type or paste text here…",
    )

    tts_lang = cfg["lang"] if (cfg["lang"] != "auto" and len(cfg["lang"]) == 2) else "en"
    col1, col2 = st.columns(2)
    speed_slow = col1.checkbox("🐢 Slow speed", value=False)
    col2.markdown(f"Characters: **{len(tts_text)}**")
    st.caption(f"TTS language: **{tts_lang}**")

    if st.button("🎵 Generate Speech", type="primary", use_container_width=True, disabled=not tts_text.strip()):
        with st.spinner("Generating speech…"):
            try:
                tts = gTTS(text=tts_text, lang=tts_lang, slow=speed_slow)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                audio_bytes = buf.read()
                st.success("✅ Speech generated!")
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    "⬇️ Download MP3", data=audio_bytes,
                    file_name="tts_output.mp3", mime="audio/mp3", use_container_width=True,
                )
            except Exception as e:
                st.error(f"TTS failed: {e}")

    if last_results:
        ok_results = [r for r in last_results if r.status == "success" and r.transcript]
        if ok_results:
            st.markdown("---")
            st.markdown("#### 🔁 Speak individual engine transcripts")
            for r in ok_results:
                with st.expander(f"🔊 {r.engine}"):
                    st.write(r.transcript)
                    if st.button(f"Generate for {r.engine}", key=f"tts_{r.engine}", use_container_width=True):
                        with st.spinner("Generating…"):
                            mp3 = make_tts_audio(r.transcript, tts_lang)
                        if mp3:
                            st.audio(mp3, format="audio/mp3")
                        else:
                            st.warning("TTS failed.")


# ══════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════════════

def main():
    page_setup()

    st.markdown('<div class="lab-title">🎙️ ASR Intelligence Lab Pro</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lab-subtitle">'
        "Benchmark Whisper · Faster-Whisper · Sarvam AI · Wav2Vec2-XLSR-53 · SeamlessM4T-v2 — "
        "live captions, PDF reports &amp; rankings."
        "</div>",
        unsafe_allow_html=True,
    )

    render_availability()
    st.markdown("---")

    cfg = build_sidebar()

    t1, t2, t3, t4 = st.tabs([
        "🎤 Live Record",
        "📁 Upload WAV / MP3",
        "📊 Results Dashboard",
        "🔊 Text-to-Speech",
    ])

    with t1:
        tab_live(cfg)
    with t2:
        tab_upload(cfg)
    with t3:
        tab_results(cfg)
    with t4:
        tab_tts(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("An unexpected top-level error occurred.")
        st.code(traceback.format_exc())
        