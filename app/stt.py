"""Локальное распознавание речи (faster-whisper) и диаризация.

Аудио никуда не отправляется: все модели скачиваются один раз и работают на этой машине.
"""
import logging
import re
import subprocess
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

_whisper = None
_ecapa = None
_pyannote = None

KAZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


# ---------- подготовка аудио ----------

def to_wav16k(src: Path) -> Path:
    """Любой аудио/видеофайл -> WAV 16 кГц моно (нужен ffmpeg)."""
    dst = src.with_suffix(".16k.wav")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg не смог обработать файл: {proc.stderr[-500:]}")
    return dst


def load_audio(wav: Path) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == 16000, "ожидается 16 кГц"
    return audio


# ---------- распознавание ----------

def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        device = config.WHISPER_DEVICE
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = config.WHISPER_COMPUTE_TYPE
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        log.info("Загрузка Whisper %s (%s, %s)", config.WHISPER_MODEL, device, compute)
        _whisper = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute,
                                download_root=str(config.MODELS_DIR / "whisper"))
    return _whisper


def detect_lang(text: str) -> str:
    """Грубая эвристика для подписи реплики: kk / ru / mixed (шала-казахский)."""
    words = re.findall(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]+", text)
    if not words:
        return "other"
    kaz = sum(1 for w in words if set(w) & KAZ_LETTERS)
    share = kaz / len(words)
    if share >= 0.35:
        return "kk"
    if kaz > 0:
        return "mixed"
    return "ru"


def transcribe(wav: Path) -> tuple[list[dict], float]:
    model = _get_whisper()
    kwargs = dict(beam_size=5, vad_filter=True, initial_prompt=config.WHISPER_PROMPT,
                  condition_on_previous_text=False)
    try:
        # multilingual=True: язык определяется для каждого фрагмента — важно для смешанной речи
        segments, info = model.transcribe(str(wav), multilingual=True, **kwargs)
    except TypeError:  # старая версия faster-whisper
        segments, info = model.transcribe(str(wav), **kwargs)
    result = []
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        result.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text,
                       "lang": detect_lang(text)})
    return result, float(getattr(info, "duration", 0.0) or 0.0)


# ---------- диаризация ----------

def _diarize_pyannote(wav: Path, segments: list[dict], num_speakers: int | None) -> None:
    global _pyannote
    from pyannote.audio import Pipeline
    if _pyannote is None:
        name = "pyannote/speaker-diarization-3.1"
        try:
            _pyannote = Pipeline.from_pretrained(name, use_auth_token=config.HF_TOKEN)
        except TypeError:
            _pyannote = Pipeline.from_pretrained(name, token=config.HF_TOKEN)
        try:
            import torch
            if torch.cuda.is_available():
                _pyannote.to(torch.device("cuda"))
        except Exception:
            pass
    params = {"num_speakers": num_speakers} if num_speakers else {}
    out = _pyannote(str(wav), **params)
    annotation = getattr(out, "speaker_diarization", out)
    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    # Каждой реплике Whisper — говорящий с максимальным пересечением по времени
    for seg in segments:
        best, best_ov = None, 0.0
        for start, end, spk in turns:
            ov = min(seg["end"], end) - max(seg["start"], start)
            if ov > best_ov:
                best, best_ov = spk, ov
        seg["speaker_raw"] = best or "UNKNOWN"


def _get_ecapa():
    global _ecapa
    if _ecapa is None:
        from speechbrain.inference.speaker import EncoderClassifier
        kwargs = dict(source="speechbrain/spkrec-ecapa-voxceleb",
                      savedir=str(config.MODELS_DIR / "ecapa"), run_opts={"device": "cpu"})
        try:
            from speechbrain.utils.fetching import LocalStrategy
            _ecapa = EncoderClassifier.from_hparams(local_strategy=LocalStrategy.COPY, **kwargs)
        except (ImportError, TypeError):
            _ecapa = EncoderClassifier.from_hparams(**kwargs)
    return _ecapa


def _diarize_ecapa(wav: Path, segments: list[dict], num_speakers: int | None) -> None:
    import torch
    from sklearn.cluster import AgglomerativeClustering

    audio = load_audio(wav)
    model = _get_ecapa()
    embeddings, idx = [], []
    for i, seg in enumerate(segments):
        a, b = int(seg["start"] * 16000), int(seg["end"] * 16000)
        chunk = audio[a:b]
        if len(chunk) < 16000 * 0.6:  # слишком короткие фрагменты дают шумный эмбеддинг
            continue
        with torch.no_grad():
            emb = model.encode_batch(torch.from_numpy(chunk).unsqueeze(0)).squeeze().cpu().numpy()
        embeddings.append(emb / (np.linalg.norm(emb) + 1e-9))
        idx.append(i)

    if len(embeddings) < 2:
        for seg in segments:
            seg["speaker_raw"] = "0"
        return

    X = np.vstack(embeddings)
    if num_speakers and num_speakers <= len(X):
        clust = AgglomerativeClustering(n_clusters=num_speakers, metric="cosine", linkage="average")
    else:
        clust = AgglomerativeClustering(n_clusters=None, distance_threshold=config.DIAR_THRESHOLD,
                                        metric="cosine", linkage="average")
    labels = clust.fit_predict(X)
    for i, lab in zip(idx, labels):
        segments[i]["speaker_raw"] = str(lab)
    # Короткие реплики без эмбеддинга — ближайшему соседу по времени
    last = str(labels[0])
    for seg in segments:
        if "speaker_raw" in seg:
            last = seg["speaker_raw"]
        else:
            seg["speaker_raw"] = last


def diarize(wav: Path, segments: list[dict], num_speakers: int | None = None) -> str:
    """Проставляет seg['speaker'] = 'Спикер 1', 'Спикер 2', ... Возвращает использованный метод."""
    method = "ecapa"
    if config.HF_TOKEN:
        try:
            _diarize_pyannote(wav, segments, num_speakers)
            method = "pyannote"
        except Exception as e:  # нет пакета/доступа — работаем без токена
            log.warning("pyannote недоступен (%s), использую ECAPA", e)
    if method == "ecapa":
        _diarize_ecapa(wav, segments, num_speakers)

    # Нумеруем говорящих в порядке первого появления
    mapping: dict[str, str] = {}
    for seg in segments:
        raw = seg.pop("speaker_raw", "0")
        if raw not in mapping:
            mapping[raw] = f"Спикер {len(mapping) + 1}"
        seg["speaker"] = mapping[raw]
    return method


# ---------- текстовый режим (для проверки без аудио) ----------

LINE_RE = re.compile(r"^\s*(?:\[?[\d:]+\]?\s*)?([^:]{1,40}):\s*(.+)$")


def parse_text_transcript(text: str) -> list[dict]:
    """Строки вида «Имя: реплика» -> сегменты. Имена становятся говорящими."""
    segments, t = [], 0.0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        speaker, utt = (m.group(1).strip(), m.group(2).strip()) if m else ("Спикер 1", line)
        dur = max(2.0, len(utt.split()) * 0.4)
        segments.append({"start": round(t, 2), "end": round(t + dur, 2), "text": utt,
                         "speaker": speaker, "lang": detect_lang(utt)})
        t += dur
    return segments
