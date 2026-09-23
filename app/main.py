"""Веб-сервер: загрузка/запись совещания, обработка, поручения, напоминания, экспорт."""
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, export, llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"
jobs: "queue.Queue[tuple]" = queue.Queue()


# ---------- конвейер обработки ----------

def _set(mid: int, status: str, note: str = "") -> None:
    db.update_meeting(mid, status=status, stage_note=note)
    log.info("Совещание %s: %s %s", mid, status, note)


def run_analysis(mid: int) -> None:
    m = db.get_meeting(mid)
    _set(mid, "analyzing", "Выделение поручений и саммари (локальная LLM)")
    result = llm.analyze(m["segments"], date.fromisoformat(m["meeting_date"]), m["speakers"])
    speakers = dict(m["speakers"])
    for label, name in result["participants"].items():
        speakers.setdefault(label, name)  # имена, заданные вручную, не перезаписываем
    db.update_meeting(mid, summary=result["summary"], decisions=result["decisions"], speakers=speakers)
    db.replace_tasks(mid, result["tasks"])
    _set(mid, "done", f"Готово: поручений — {len(result['tasks'])}")


def process_audio(mid: int, path: Path, num_speakers: int | None) -> None:
    from . import stt  # тяжёлые импорты — только когда нужны
    _set(mid, "converting", "Подготовка аудио")
    wav = stt.to_wav16k(path)
    _set(mid, "transcribing", f"Распознавание речи ({config.WHISPER_MODEL}), это может занять несколько минут")
    segments, duration = stt.transcribe(wav)
    _set(mid, "diarizing", "Определение говорящих")
    method = stt.diarize(wav, segments, num_speakers) if segments else "none"
    db.update_meeting(mid, segments=segments, diarization_method=method, duration_sec=duration)
    run_analysis(mid)


def worker() -> None:
    while True:
        kind, mid, *args = jobs.get()
        try:
            if kind == "audio":
                process_audio(mid, *args)
            elif kind == "analyze":
                run_analysis(mid)
        except Exception as e:
            log.exception("Ошибка обработки совещания %s", mid)
            db.update_meeting(mid, status="error", error=str(e)[:1000], stage_note="Ошибка")
        finally:
            jobs.task_done()


# ---------- напоминания (сценарий 2) ----------

def reminder_loop() -> None:
    while True:
        try:
            for t in db.list_tasks():
                state = t["state"]
                who = t["assignee_display"]
                if state == "overdue" and t.get("notified") != "overdue":
                    db.add_notification(t["id"], "overdue",
                                        f"Просрочено: «{t['task']}» — {who}, срок {t['deadline']}")
                elif state == "due_soon" and t.get("notified") not in ("due_soon", "overdue"):
                    db.add_notification(t["id"], "due_soon",
                                        f"Скоро срок: «{t['task']}» — {who}, до {t['deadline']}")
        except Exception:
            log.exception("Ошибка проверки сроков")
        time.sleep(config.REMINDER_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=reminder_loop, daemon=True).start()
    if not config.llm_is_local():
        log.warning("LLM_BASE_URL указывает на внешний адрес — это нарушает требование закрытого контура")
    yield


app = FastAPI(title="Автопротокол совещаний", lifespan=lifespan)


# ---------- API ----------

@app.get("/api/health")
def health():
    return {"whisper_model": config.WHISPER_MODEL, "llm": llm.check_llm(),
            "diarization": "pyannote" if config.HF_TOKEN else "ecapa", "queue": jobs.qsize()}


@app.post("/api/meetings")
async def create_from_audio(file: UploadFile = File(...), title: str = Form(...),
                            meeting_date: str = Form(...), consent: bool = Form(False),
                            num_speakers: int | None = Form(None)):
    if not consent:
        raise HTTPException(400, "Подтвердите, что участники уведомлены о записи и ИИ-транскрибации")
    _validate_date(meeting_date)
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    path = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    mid = db.create_meeting(title.strip() or "Совещание", meeting_date, file.filename or "запись", consent)
    db.update_meeting(mid, audio_path=str(path))
    jobs.put(("audio", mid, path, num_speakers if num_speakers and num_speakers > 0 else None))
    return {"id": mid}


class TextMeeting(BaseModel):
    title: str
    meeting_date: str
    transcript: str


@app.post("/api/meetings/text")
def create_from_text(body: TextMeeting):
    """Проверка без аудио: стенограмма строками «Имя: реплика»."""
    from .stt import parse_text_transcript
    _validate_date(body.meeting_date)
    segments = parse_text_transcript(body.transcript)
    if not segments:
        raise HTTPException(400, "Пустая стенограмма")
    mid = db.create_meeting(body.title.strip() or "Совещание", body.meeting_date, "текст", True)
    db.update_meeting(mid, segments=segments, diarization_method="text")
    jobs.put(("analyze", mid))
    return {"id": mid}


@app.get("/api/meetings")
def meetings():
    return db.list_meetings()


@app.get("/api/meetings/{mid}")
def meeting(mid: int):
    return _meeting_or_404(mid)


AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
               ".webm": "audio/webm", ".mp4": "video/mp4", ".opus": "audio/ogg", ".flac": "audio/flac"}


@app.get("/api/meetings/{mid}/audio")
def meeting_audio(mid: int):
    """Исходная запись для прослушивания рядом со стенограммой (отдаётся только с этого сервера)."""
    path = db.get_audio_path(mid)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Запись недоступна")
    return FileResponse(path, media_type=AUDIO_TYPES.get(Path(path).suffix.lower(), "application/octet-stream"))


@app.delete("/api/meetings/{mid}")
def remove_meeting(mid: int):
    _meeting_or_404(mid)
    db.delete_meeting(mid)
    return {"ok": True}


@app.put("/api/meetings/{mid}/speakers")
def rename_speakers(mid: int, names: dict[str, str]):
    m = _meeting_or_404(mid)
    speakers = dict(m["speakers"])
    for label, name in names.items():
        if name.strip():
            speakers[label] = name.strip()
        else:
            speakers.pop(label, None)
    db.update_meeting(mid, speakers=speakers)
    return db.get_meeting(mid)


@app.post("/api/meetings/{mid}/reanalyze")
def reanalyze(mid: int):
    _meeting_or_404(mid)
    jobs.put(("analyze", mid))
    db.update_meeting(mid, status="queued", stage_note="В очереди на повторный анализ")
    return {"ok": True}


@app.get("/api/meetings/{mid}/export")
def export_meeting(mid: int, format: str = "docx"):
    m = _meeting_or_404(mid)
    if m["status"] != "done":
        raise HTTPException(409, "Протокол ещё не готов")
    if format == "pdf":
        path, mime = export.export_pdf(m), "application/pdf"
    elif format == "docx":
        path = export.export_docx(m)
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        raise HTTPException(400, "Формат: pdf или docx")
    return FileResponse(path, media_type=mime, filename=f"protocol_{mid}.{format}")


class TaskPatch(BaseModel):
    task: str | None = None
    assignee: str | None = None
    assignee_speaker: str | None = None
    deadline: str | None = None
    priority: str | None = None
    status: str | None = None


@app.get("/api/tasks")
def tasks():
    return db.list_tasks()


@app.patch("/api/tasks/{tid}")
def patch_task(tid: int, body: TaskPatch):
    if not db.get_task(tid):
        raise HTTPException(404, "Поручение не найдено")
    fields = body.model_dump(exclude_unset=True)
    if fields.get("status") not in (None, "in_progress", "done"):
        raise HTTPException(400, "Статус: in_progress или done")
    if fields.get("deadline"):
        _validate_date(fields["deadline"])
    db.update_task(tid, **fields)
    return db.get_task(tid)


@app.delete("/api/tasks/{tid}")
def remove_task(tid: int):
    db.delete_task(tid)
    return {"ok": True}


@app.get("/api/notifications")
def notifications():
    return db.list_notifications()


def _meeting_or_404(mid: int) -> dict:
    m = db.get_meeting(mid)
    if not m:
        raise HTTPException(404, "Совещание не найдено")
    return m


def _validate_date(value: str) -> None:
    try:
        date.fromisoformat(value)
    except ValueError:
        raise HTTPException(400, "Дата должна быть в формате ГГГГ-ММ-ДД")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
