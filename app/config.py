"""Настройки приложения. Все значения берутся из .env (см. .env.example)."""
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "meetings.db"
for d in (DATA_DIR, UPLOAD_DIR, EXPORT_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Распознавание речи (локально, faster-whisper) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")          # auto | cpu | cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")  # auto | int8 | float16
# Подсказка модели: помогает с казахской/русской лексикой совещаний и смешанной речью
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "Совещание. Жиналыс. Поручение, срок, ответственный. Тапсырма, мерзімі, жауапты. "
    "Жарайды, коллеги, давайте бастайық.",
)

# --- Диаризация (локально) ---
# Если задан HF_TOKEN и установлен pyannote.audio — используется pyannote,
# иначе — эмбеддинги ECAPA (SpeechBrain) + кластеризация (токен не нужен).
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
DIAR_THRESHOLD = float(os.getenv("DIAR_THRESHOLD", "0.65"))

# --- LLM для извлечения поручений (локально, Ollama или любой OpenAI-совместимый self-hosted сервер) ---
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")               # ollama | openai_compatible
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "900"))
# Защита закрытого контура: запрос к нелокальному LLM блокируется, пока явно не разрешён.
ALLOW_EXTERNAL_LLM = os.getenv("ALLOW_EXTERNAL_LLM", "0") == "1"

# --- Напоминания ---
REMIND_DAYS_BEFORE = int(os.getenv("REMIND_DAYS_BEFORE", "1"))
REMINDER_INTERVAL_SEC = int(os.getenv("REMINDER_INTERVAL_SEC", "60"))

FONTS_DIR = BASE_DIR / "app" / "fonts"


def _is_private_host(host: str) -> bool:
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal", "ollama"):
        return True
    if host.startswith(("10.", "192.168.")):
        return True
    if host.startswith("172."):
        try:
            return 16 <= int(host.split(".")[1]) <= 31
        except (IndexError, ValueError):
            return False
    return host.endswith((".local", ".lan", ".internal"))


def llm_is_local() -> bool:
    return _is_private_host(urlparse(LLM_BASE_URL).hostname or "")
