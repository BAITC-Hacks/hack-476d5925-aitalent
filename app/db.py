"""Хранилище на SQLite: совещания, поручения, уведомления."""
import json
import sqlite3
import threading
from datetime import date, datetime

from .config import DB_PATH, REMIND_DAYS_BEFORE

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    meeting_date TEXT NOT NULL,
    source TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    stage_note TEXT,
    error TEXT,
    consent INTEGER NOT NULL DEFAULT 0,
    segments_json TEXT,
    speakers_json TEXT,
    summary TEXT,
    decisions_json TEXT,
    diarization_method TEXT,
    duration_sec REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    assignee TEXT,
    assignee_speaker TEXT,
    deadline TEXT,
    deadline_text TEXT,
    priority TEXT,
    category TEXT,
    quote TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress',
    notified TEXT
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------- совещания ----------

def create_meeting(title: str, meeting_date: str, source: str, consent: bool) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO meetings (title, meeting_date, source, consent, created_at) VALUES (?,?,?,?,?)",
            (title, meeting_date, source, int(consent), _now()),
        )
        return cur.lastrowid


def update_meeting(meeting_id: int, **fields) -> None:
    for key in ("segments", "speakers", "decisions"):
        if key in fields:
            fields[f"{key}_json"] = json.dumps(fields.pop(key), ensure_ascii=False)
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE meetings SET {cols} WHERE id = ?", (*fields.values(), meeting_id))


def _meeting_row(row: sqlite3.Row) -> dict:
    m = dict(row)
    m["segments"] = json.loads(m.pop("segments_json") or "[]")
    m["speakers"] = json.loads(m.pop("speakers_json") or "{}")
    m["decisions"] = json.loads(m.pop("decisions_json") or "[]")
    m["consent"] = bool(m["consent"])
    return m


def get_meeting(meeting_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    if not row:
        return None
    m = _meeting_row(row)
    m["tasks"] = list_tasks(meeting_id=meeting_id)
    return m


def list_meetings() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT m.id, m.title, m.meeting_date, m.status, m.stage_note, m.created_at, "
            "(SELECT COUNT(*) FROM tasks t WHERE t.meeting_id = m.id) AS task_count "
            "FROM meetings m ORDER BY m.id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_meeting(meeting_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))


# ---------- поручения ----------

def replace_tasks(meeting_id: int, tasks: list[dict]) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM tasks WHERE meeting_id = ?", (meeting_id,))
        for t in tasks:
            c.execute(
                "INSERT INTO tasks (meeting_id, task, assignee, assignee_speaker, deadline, deadline_text,"
                " priority, category, quote) VALUES (?,?,?,?,?,?,?,?,?)",
                (meeting_id, t.get("task", ""), t.get("assignee"), t.get("assignee_speaker"),
                 t.get("deadline"), t.get("deadline_text"), t.get("priority"),
                 t.get("category"), t.get("quote")),
            )


def task_state(task: dict, today: date | None = None) -> str:
    """Статус для дашборда: done | overdue | due_soon | in_progress."""
    if task["status"] == "done":
        return "done"
    if not task.get("deadline"):
        return "in_progress"
    today = today or date.today()
    try:
        d = date.fromisoformat(task["deadline"])
    except ValueError:
        return "in_progress"
    if d < today:
        return "overdue"
    if (d - today).days <= REMIND_DAYS_BEFORE:
        return "due_soon"
    return "in_progress"


def list_tasks(meeting_id: int | None = None) -> list[dict]:
    sql = ("SELECT t.*, m.title AS meeting_title, m.meeting_date, m.speakers_json "
           "FROM tasks t JOIN meetings m ON m.id = t.meeting_id")
    args: tuple = ()
    if meeting_id is not None:
        sql += " WHERE t.meeting_id = ?"
        args = (meeting_id,)
    sql += " ORDER BY (t.deadline IS NULL), t.deadline, t.id"
    with _conn() as c:
        rows = c.execute(sql, args).fetchall()
    result = []
    for r in rows:
        t = dict(r)
        speakers = json.loads(t.pop("speakers_json") or "{}")
        # Если говорящему присвоено имя — показываем его как ответственного
        named = speakers.get(t.get("assignee_speaker") or "")
        t["assignee_display"] = named or t.get("assignee") or t.get("assignee_speaker") or "не указан"
        t["state"] = task_state(t)
        result.append(t)
    return result


def get_task(task_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def update_task(task_id: int, **fields) -> None:
    allowed = {"task", "assignee", "assignee_speaker", "deadline", "priority", "category", "status"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if "deadline" in fields:
        fields["notified"] = None  # срок изменился — напоминание нужно заново
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE tasks SET {cols} WHERE id = ?", (*fields.values(), task_id))


def delete_task(task_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


# ---------- уведомления ----------

def add_notification(task_id: int, kind: str, message: str) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT INTO notifications (task_id, kind, message, created_at) VALUES (?,?,?,?)",
                  (task_id, kind, message, _now()))
        c.execute("UPDATE tasks SET notified = ? WHERE id = ?", (kind, task_id))


def list_notifications(limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
