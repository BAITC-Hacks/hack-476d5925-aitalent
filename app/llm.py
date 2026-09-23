"""Извлечение поручений и саммари локальной LLM (Ollama или self-hosted OpenAI-совместимый сервер)."""
import json
import logging
import re
from datetime import date, timedelta

import httpx

from . import config

log = logging.getLogger(__name__)

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]

SYSTEM_PROMPT = """Ты — ассистент секретаря совещаний в государственной организации Казахстана.
Тебе дают транскрипт совещания. Речь может быть на русском, казахском или смешанной («шала-казахский»).
Каждая строка: [время] Говорящий: реплика. Говорящие обозначены как «Спикер N» или по имени.

Твоя задача — вернуть СТРОГО один JSON-объект без пояснений и без markdown:
{
  "summary": "краткое саммари совещания на русском, 3–6 предложений",
  "decisions": ["принятое решение 1", "..."],
  "participants": {"Спикер 1": "имя, если оно прозвучало в разговоре, иначе null"},
  "tasks": [
    {
      "task": "суть поручения на русском, начиная с глагола (Подготовить..., Направить...)",
      "assignee": "имя ответственного так, как оно прозвучало, или null",
      "assignee_speaker": "метка говорящего-исполнителя (например, «Спикер 2») или null",
      "deadline_text": "срок дословно, как сказали («до пятницы», «жұмаға дейін»), или null",
      "deadline": "срок в формате YYYY-MM-DD или null",
      "priority": "высокий | средний | низкий",
      "category": "краткое направление: финансы, ИТ, кадры, документы, закупки и т.п.",
      "quote": "короткая цитата из транскрипта, где дано поручение"
    }
  ]
}

Правила:
- Поручение — это конкретное действие, которое кому-то поручили или которое кто-то взял на себя. Общие обсуждения не являются поручениями.
- Исполнитель: обычно это тот, к кому обращаются («Ерлан, подготовь…») или кто отвечает согласием («Хорошо, сделаю», «Жарайды, орындаймын»). Определи его метку говорящего по контексту диалога.
- Если имя человека звучит в обращении, а следующим отвечает определённый говорящий, свяжи это имя с его меткой в "participants" и "assignee_speaker".
- Срок вычисляй относительно даты совещания. Казахские слова: дүйсенбі=понедельник, сейсенбі=вторник, сәрсенбі=среда, бейсенбі=четверг, жұма=пятница, сенбі=суббота, жексенбі=воскресенье, ертең=завтра, апта=неделя, ай=месяц. «До пятницы» = ближайшая пятница после даты совещания.
- Если срок не назван — deadline и deadline_text = null. Не выдумывай сроки, имена и поручения.
- Приоритет «высокий», если сказано «срочно», «шұғыл», «в первую очередь» или срок ≤ 2 дней.
- Все тексты в ответе — на русском языке, кроме поля quote (оставь как в оригинале)."""


def _user_prompt(transcript: str, meeting_date: date) -> str:
    return (f"Дата совещания: {meeting_date.isoformat()} ({WEEKDAYS[meeting_date.weekday()]}).\n\n"
            f"Транскрипт:\n{transcript}")


def format_transcript(segments: list[dict], speaker_names: dict | None = None) -> str:
    speaker_names = speaker_names or {}
    lines = []
    for s in segments:
        m, sec = divmod(int(s["start"]), 60)
        spk = s.get("speaker", "Спикер")
        name = speaker_names.get(spk)
        label = f"{spk} ({name})" if name and name != spk else spk
        lines.append(f"[{m:02d}:{sec:02d}] {label}: {s['text']}")
    return "\n".join(lines)


def _chat(messages: list[dict]) -> str:
    if not config.llm_is_local() and not config.ALLOW_EXTERNAL_LLM:
        raise RuntimeError(
            f"LLM_BASE_URL={config.LLM_BASE_URL} не является локальным адресом. Передача текста совещаний "
            "во внешние сервисы запрещена требованиями закрытого контура. Используйте локальный Ollama/vLLM."
        )
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
    with httpx.Client(timeout=config.LLM_TIMEOUT) as client:
        if config.LLM_BACKEND == "ollama":
            r = client.post(f"{config.LLM_BASE_URL}/api/chat", headers=headers, json={
                "model": config.LLM_MODEL, "messages": messages, "stream": False, "format": "json",
                "options": {"temperature": 0.1, "num_ctx": config.LLM_NUM_CTX},
            })
            r.raise_for_status()
            return r.json()["message"]["content"]
        # OpenAI-совместимый self-hosted сервер (vLLM, NVIDIA NIM on-prem, LM Studio и т.п.)
        base = config.LLM_BASE_URL if config.LLM_BASE_URL.endswith("/v1") else f"{config.LLM_BASE_URL}/v1"
        r = client.post(f"{base}/chat/completions", headers=headers, json={
            "model": config.LLM_MODEL, "messages": messages, "temperature": 0.1,
            "response_format": {"type": "json_object"},
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _clean_date(value, meeting_date: date) -> str | None:
    if not value or not isinstance(value, str):
        return None
    try:
        d = date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
    # отбрасываем явно ошибочные даты (в прошлом относительно совещания или через годы)
    if d < meeting_date or d > meeting_date + timedelta(days=730):
        return None
    return d.isoformat()


def analyze(segments: list[dict], meeting_date: date, speaker_names: dict | None = None) -> dict:
    transcript = format_transcript(segments, speaker_names)
    if not transcript.strip():
        return {"summary": "Речь в записи не распознана.", "decisions": [], "participants": {}, "tasks": []}

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(transcript, meeting_date)}]
    raw = _chat(messages)
    try:
        data = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("LLM вернула невалидный JSON, повторяю запрос")
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "Ответ не является валидным JSON. Верни только JSON-объект."}]
        data = _parse_json(_chat(messages))

    known_speakers = {s.get("speaker") for s in segments}
    tasks = []
    for t in data.get("tasks") or []:
        if not isinstance(t, dict) or not (t.get("task") or "").strip():
            continue
        spk = t.get("assignee_speaker")
        tasks.append({
            "task": t["task"].strip(),
            "assignee": (t.get("assignee") or None),
            "assignee_speaker": spk if spk in known_speakers else None,
            "deadline": _clean_date(t.get("deadline"), meeting_date),
            "deadline_text": t.get("deadline_text") or None,
            "priority": t.get("priority") if t.get("priority") in ("высокий", "средний", "низкий") else "средний",
            "category": t.get("category") or None,
            "quote": t.get("quote") or None,
        })

    participants = {k: v for k, v in (data.get("participants") or {}).items()
                    if k in known_speakers and isinstance(v, str) and v.strip() and v.lower() != "null"}
    return {
        "summary": (data.get("summary") or "").strip(),
        "decisions": [d for d in (data.get("decisions") or []) if isinstance(d, str) and d.strip()],
        "participants": participants,
        "tasks": tasks,
    }


def check_llm() -> dict:
    """Проверка доступности LLM для страницы статуса."""
    info = {"backend": config.LLM_BACKEND, "model": config.LLM_MODEL, "url": config.LLM_BASE_URL,
            "local": config.llm_is_local(), "reachable": False}
    try:
        with httpx.Client(timeout=3) as c:
            path = "/api/tags" if config.LLM_BACKEND == "ollama" else "/v1/models"
            info["reachable"] = c.get(config.LLM_BASE_URL.removesuffix("/v1") + path).status_code < 500
    except Exception:
        pass
    return info
