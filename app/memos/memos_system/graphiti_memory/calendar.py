"""Rule-only calendar view: the transcript slice that carries a time expression.

The fact graph stores what happened, never *when it was said*: "上周三我们说了什么"
names a day, and no object in the graph is addressable by that day. This module
keeps the small slice of transcript that carries a time expression, resolves the
expression to concrete day(s) with pure rules (no LLM call, no extraction), and
lets recall address it by day.

A view, not authority (ADR-0178):

* rows are derived from a recorded turn and can be replayed from it;
* nothing here is a Graphiti extraction source, so it cannot pollute entity
  dedupe, fact merging or the publication saga;
* the store is append-only per role and is deleted with the role.

Recall reads it two ways:

* **by day** — the query itself carries a time expression ("上周三", "last Friday"),
  and rows whose resolved day intersects the query days are returned;
* **by content** — the query is a time *question* ("什么时候", "when did …") and rows
  that lexically match are returned, each still rendered with its day.

Both occupy the caller's ``top_k`` budget instead of being appended on top of it:
the calendar must not silently enlarge what the model sees.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import score_text, utc_timestamp

#: A sentence longer than this is not a calendar line any more; it is a paragraph
#: that happened to contain "明天".
_MAX_SENTENCE_CHARS = 300
#: One turn contributes at most this many rows, so a long imported transcript
#: cannot flood the view in a single append.
_MAX_ROWS_PER_TURN = 6
#: A single month-level expression resolves to every day of that month; capping
#: the expansion keeps a stored row small.
_MAX_DAYS_PER_HIT = 31

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

_DAY_WORDS = {
    "今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3,
    "昨天": -1, "昨日": -1, "前天": -2, "大前天": -3, "当天": 0, "当日": 0,
}

_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_EN_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6}
_EN_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
              "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
              "november": 11, "december": 12,
              "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
              "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}

#: Order matters: a longer phrase must win the span before a shorter one re-reads
#: part of it ("10月13日" before "13日", "大后天" before "后天").
_DAY_WORD_RE = re.compile("|".join(sorted(_DAY_WORDS, key=len, reverse=True)))
_OFFSET_RE = re.compile(
    r"(?P<n>[0-9]{1,3}|[一二三四五六七八九十两]{1,3})\s*(?P<unit>天|日|个?月|年)\s*"
    r"(?P<dir>以后|之后|过后|后|以前|之前|前)"
)
_ABS_MONTH_DAY_RE = re.compile(
    r"(?<![0-9])(?P<m>[0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*月\s*"
    r"(?P<d>[0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*[日号]"
)
_ABS_MONTH_RE = re.compile(
    r"(?<![0-9])(?P<m>[0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*月(?!\s*[日号底初中])"
)
_DAY_OF_MONTH_RE = re.compile(r"(?<![0-9月])(?P<d>[0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*[日号]")
_WEEKDAY_RE = re.compile(
    r"(?P<rel>这|本|下|上)?\s*(?:周|星期|礼拜)(?:\s*(?:周|星期|礼拜))?\s*(?P<w>[一二三四五六日天])"
)
_MONTH_REL_RE = re.compile(
    r"(?:(?P<nm>[0-9]{1,2}|[一二三四五六七八九十]{1,3})\s*月(?P<ne>底|初))"
    r"|(?:(?P<rel>这个?月|下个?月|上个?月|下下个?月|月底|月初)(?P<e>底|初)?)"
)
_CLOCK_RE = re.compile(
    r"(?P<c>[0-9]{1,2}\s*[:：]\s*[0-9]{2}"
    r"|[0-9一二三四五六七八九十]{1,3}\s*点\s*(?:半|一刻|钟)?)"
)
_PART_OF_DAY_RE = re.compile(r"(早上|上午|中午|下午|傍晚|晚上|夜里|夜间|凌晨|清早|半夜)")

#: "一点都不合理" / "略大一点" are not clock times. The pronoun/degree verbs that
#: precede a numeric "一点" are the give-away, and a following 都/也/儿/不 pins it.
_CLOCK_FALSE_PREFIX = re.compile(
    r"(有|差|快|慢|多|少|这|那|哪|稍微|略|一点|[大小高低长短早晚好])$"
)
_CLOCK_FALSE_SUFFIX = re.compile(r"^\s*(都|也|儿|不|点)")

_EN_DAY_WORDS = {"today": 0, "yesterday": -1, "tomorrow": 1, "tonight": 0}
#: Longer English idioms first: "the day before yesterday" must not be read as
#: plain "yesterday".
_EN_IDIOM_RE = re.compile(
    r"\bthe day (?P<dir>before|after) (?P<anchor>yesterday|tomorrow)\b", re.I
)
_EN_DAY_RE = re.compile(r"\b(?P<w>today|yesterday|tomorrow|tonight)\b", re.I)
_EN_REL_RE = re.compile(
    r"\b(?P<rel>last|next|this|past)\s+(?P<unit>week|month|year|weekend)\b", re.I
)
_EN_AGO_RE = re.compile(
    r"\b(?P<n>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>day|week|month|year)s?\s+(?P<dir>ago|later|from now)\b", re.I
)
_EN_WEEKDAY_RE = re.compile(
    r"\b(?:(?P<rel>last|next|this|past)\s+)?(?P<w>monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b", re.I
)
_EN_MONTH_DAY_RE = re.compile(
    r"\b(?P<mon>january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+"
    r"(?P<d>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{4}))?\b", re.I
)
_EN_MONTH_YEAR_RE = re.compile(
    r"\b(?P<mon>january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?,?\s+"
    r"(?P<y>\d{4})\b", re.I
)

_TIME_QUESTION_RE = re.compile(
    r"(什么时候|啥时候|哪天|哪一天|几号|几月|几点|多久|多长时间|相隔多久|"
    r"\bwhen\b|\bwhat (?:date|day|time)\b|\bhow long\b|\bwhich (?:date|day)\b)",
    re.I,
)


def _cn_number(text: str) -> Optional[int]:
    """Parse ``13``, ``十三``, ``二十三``, ``两`` into an int."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in _CN_DIGITS:
        return _CN_DIGITS[raw]
    if "十" in raw:
        head, _, tail = raw.partition("十")
        tens = _CN_DIGITS.get(head, 1) if head else 1
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        if (head and head not in _CN_DIGITS) or (tail and tail not in _CN_DIGITS):
            return None
        return tens * 10 + ones
    return None


def _add_months(day: date, months: int) -> date:
    total = day.month - 1 + months
    year = day.year + total // 12
    month = total % 12 + 1
    # Clamp to the last day of the target month (31 Mar - 1 month = 28/29 Feb).
    for candidate in (31, 30, 29, 28):
        try:
            return date(year, month, min(day.day, candidate))
        except ValueError:
            continue
    return day


def _month_days(year: int, month: int) -> Tuple[date, ...]:
    start = date(year, month, 1)
    end = _add_months(start, 1)
    return tuple(start + timedelta(days=i) for i in range((end - start).days))[:_MAX_DAYS_PER_HIT]


def _week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _month_day(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    month = _cn_number(match.group("m"))
    day_of_month = _cn_number(match.group("d"))
    if not month or not day_of_month or not (1 <= month <= 12) or not (1 <= day_of_month <= 31):
        return ()
    year = ref.year
    try:
        candidate = date(year, month, day_of_month)
    except ValueError:
        return ()
    # A named date is almost always the most recent one in the past; a future
    # year would make every "10月13日" point at a day that never happened yet.
    if candidate > ref:
        try:
            candidate = date(year - 1, month, day_of_month)
        except ValueError:
            return ()
    return (candidate,)


def _english_month_day(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    month = _EN_MONTHS.get(match.group("mon").lower().rstrip("."))
    if not month:
        return ()
    day_of_month = _cn_number(match.group("d"))
    if not day_of_month or not (1 <= day_of_month <= 31):
        return ()
    year_text = match.group("y")
    if year_text:
        try:
            return (date(int(year_text), month, day_of_month),)
        except ValueError:
            return ()
    return _month_day_from(month, day_of_month, ref)


def _month_day_from(month: int, day_of_month: int, ref: date) -> Tuple[date, ...]:
    try:
        candidate = date(ref.year, month, day_of_month)
    except ValueError:
        return ()
    if candidate > ref:
        try:
            candidate = date(ref.year - 1, month, day_of_month)
        except ValueError:
            return ()
    return (candidate,)


def _weekday_day(weekday: int, rel: Optional[str], ref: date) -> Tuple[date, ...]:
    start = _week_start(ref)
    if rel in {"下", "next"}:
        return (start + timedelta(days=7 + weekday),)
    if rel in {"上", "last", "past"}:
        return (start + timedelta(days=-7 + weekday),)
    if rel in {"这", "本", "this"}:
        return (start + timedelta(days=weekday),)
    # No qualifier: the most recent such weekday (today counts).
    delta = (ref.weekday() - weekday) % 7
    return (ref - timedelta(days=delta),)


def _bare_month(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    """A bare month ("10月") with no day: the whole month of the reference year."""
    month = _cn_number(match.group("m"))
    if not month or not 1 <= month <= 12:
        return ()
    return _month_days(ref.year, month)


def _day_of_month(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    """A bare day ("13号") with no month: this month, or the previous one."""
    day_of_month = _cn_number(match.group("d"))
    if not day_of_month or not 1 <= day_of_month <= 31:
        return ()
    try:
        candidate = date(ref.year, ref.month, day_of_month)
    except ValueError:
        return ()
    if candidate > ref:
        previous = _add_months(date(ref.year, ref.month, 1), -1)
        try:
            candidate = date(previous.year, previous.month, day_of_month)
        except ValueError:
            return ()
    return (candidate,)


def _month_rel(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    rel = str(match.group("rel") or "")
    edge = str(match.group("ne") or match.group("e") or "")
    explicit = _cn_number(match.group("nm")) if match.group("nm") else None
    if explicit and 1 <= explicit <= 12:
        base = date(ref.year, explicit, 1)
    elif rel.startswith("下下"):
        base = _add_months(date(ref.year, ref.month, 1), 2)
    elif rel.startswith("下"):
        base = _add_months(date(ref.year, ref.month, 1), 1)
    elif rel.startswith("上"):
        base = _add_months(date(ref.year, ref.month, 1), -1)
    else:
        base = date(ref.year, ref.month, 1)
    if edge == "底" or rel.endswith("底"):
        return (_add_months(base, 1) - timedelta(days=1),)
    if edge == "初" or rel.endswith("初"):
        return (base,)
    return _month_days(base.year, base.month)


def _offset(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    count = _cn_number(match.group("n"))
    if count is None:
        return ()
    sign = -1 if match.group("dir") in {"以前", "之前", "前"} else 1
    unit = match.group("unit")
    if unit in {"天", "日"}:
        return (ref + timedelta(days=sign * count),)
    if unit in {"月", "个月"}:
        return (_add_months(ref, sign * count),)
    try:
        return (date(ref.year + sign * count, ref.month, ref.day),)
    except ValueError:
        return ()


def _english_relative(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    rel = match.group("rel").lower()
    unit = match.group("unit").lower()
    sign = -1 if rel == "last" else 1
    if unit == "weekend":
        # A weekend is not one day; the Saturday of that week is the anchor.
        base = _week_start(ref) + timedelta(days=5)
        return (base + timedelta(days=7 * sign),)
    if unit == "week":
        base = _week_start(ref)
        return tuple(base + timedelta(days=7 * sign + i) for i in range(7))
    if unit == "month":
        base = _add_months(date(ref.year, ref.month, 1), sign)
        return _month_days(base.year, base.month)
    try:
        return (date(ref.year + sign, ref.month, ref.day),)
    except ValueError:
        return ()


def _english_ago(match: "re.Match[str]", ref: date) -> Tuple[date, ...]:
    raw = match.group("n").lower()
    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    count = int(raw) if raw.isdigit() else words.get(raw)
    if count is None:
        return ()
    sign = -1 if match.group("dir").lower() == "ago" else 1
    unit = match.group("unit").lower()
    if unit == "day":
        return (ref + timedelta(days=sign * count),)
    if unit == "week":
        return (ref + timedelta(days=sign * count * 7),)
    if unit == "month":
        return (_add_months(ref, sign * count),)
    try:
        return (date(ref.year + sign * count, ref.month, ref.day),)
    except ValueError:
        return ()


#: (kind, pattern, resolver). Order defines span precedence.
def _rules() -> List[Tuple[str, "re.Pattern[str]", Any]]:
    return [
        ("date-word", _DAY_WORD_RE, lambda m, ref: (ref + timedelta(days=_DAY_WORDS[m.group(0)]),)),
        ("offset", _OFFSET_RE, _offset),
        ("abs-month-day", _ABS_MONTH_DAY_RE, _month_day),
        ("abs-month", _ABS_MONTH_RE, _bare_month),
        ("day-of-month", _DAY_OF_MONTH_RE, _day_of_month),
        ("weekday", _WEEKDAY_RE, lambda m, ref: _weekday_day(_WEEKDAYS[m.group("w")], m.group("rel"), ref)),
        ("month-rel", _MONTH_REL_RE, _month_rel),
        ("en-idiom", _EN_IDIOM_RE, lambda m, ref: (ref + timedelta(
            days=(-2 if m.group("dir").lower() == "before" else 2)),)),
        ("en-date-word", _EN_DAY_RE, lambda m, ref: (ref + timedelta(days=_EN_DAY_WORDS[m.group("w").lower()]),)),
        ("en-relative", _EN_REL_RE, _english_relative),
        ("en-ago", _EN_AGO_RE, _english_ago),
        ("en-weekday", _EN_WEEKDAY_RE, lambda m, ref: _weekday_day(
            _EN_WEEKDAYS[m.group("w").lower()], (m.group("rel") or "").lower() or None, ref)),
        ("en-month-day", _EN_MONTH_DAY_RE, _english_month_day),
        ("en-month-year", _EN_MONTH_YEAR_RE, lambda m, ref: _month_days(
            int(m.group("y")), _EN_MONTHS[m.group("mon").lower().rstrip(".")])),
        ("clock", _CLOCK_RE, lambda m, ref: ()),
        ("part-of-day", _PART_OF_DAY_RE, lambda m, ref: ()),
    ]


def _clock_is_real(text: str, start: int, end: int) -> bool:
    """False for "一点都不合理"/"大一点" — a degree adverb, not a clock time."""
    before = text[max(0, start - 3):start]
    after = text[end:end + 2]
    if _CLOCK_FALSE_SUFFIX.search(after):
        return False
    if _CLOCK_FALSE_PREFIX.search(before):
        return False
    return True


def sentences(text: str) -> List[str]:
    """Split a turn body into sentences, keeping the words but not the separators."""
    parts = re.split(r"(?<=[。！？!?\n;；])", str(text or ""))
    result: List[str] = []
    for part in parts:
        cleaned = part.strip().strip("。！？!?;；").strip()
        if cleaned:
            result.append(cleaned[:_MAX_SENTENCE_CHARS])
    return result


def temporal_hits(text: str, reference: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """All time expressions in ``text``, resolved against ``reference``.

    Each hit is ``{"kind", "span", "days"}``; ``days`` is empty for an
    expression that names a time of day but no day ("12点半").
    """
    ref_dt = reference or datetime.now()
    ref = ref_dt.date()
    body = str(text or "")
    hits: List[Dict[str, Any]] = []
    taken: List[Tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and other_start < end for other_start, other_end in taken)

    for kind, pattern, resolve in _rules():
        for match in pattern.finditer(body):
            start, end = match.span()
            if start >= end or overlaps(start, end):
                continue
            if kind == "clock" and not _clock_is_real(body, start, end):
                continue
            try:
                days = tuple(resolve(match, ref))
            except (KeyError, ValueError):
                continue
            days = tuple(sorted({item for item in days if isinstance(item, date)}))[:_MAX_DAYS_PER_HIT]
            taken.append((start, end))
            hits.append({"kind": kind, "span": body[start:end].strip(), "days": days})
    hits.sort(key=lambda item: body.find(item["span"]))
    return hits


def parse_moment(value: Any) -> Optional[datetime]:
    """Accept an ISO string, an epoch number or a datetime; None when unusable.

    A replayed conversation (an import, or a benchmark that answers questions
    about 2023) is recorded long after it happened, so "今天" in its transcript
    must resolve against the *turn* time, not against wall clock.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None

def query_days(query: str, reference: Optional[datetime] = None) -> List[str]:
    """The concrete days a query asks about, as ISO strings (empty when none)."""
    days: set[str] = set()
    for hit in temporal_hits(query, reference):
        for item in hit["days"]:
            days.add(item.isoformat())
    return sorted(days)


def is_time_question(query: str) -> bool:
    """True for "什么时候/几点/when did …" — a question whose answer is a time."""
    return bool(_TIME_QUESTION_RE.search(str(query or "")))


def _row_id(role_uuid: str, conversation_id: str, day: str, sentence: str) -> str:
    raw = "\x1f".join([role_uuid, conversation_id, day, sentence])
    return "calendar:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class CalendarStore:
    """Append-only per-role calendar lines (ADR-0178)."""

    MAX_ENTRIES_PER_ROLE = 20000

    def __init__(self, root: Path | str, max_entries: Optional[int] = None):
        self.root = Path(root) / "calendar"
        self.max_entries = max(1, int(max_entries or self.MAX_ENTRIES_PER_ROLE))
        # role_uuid -> (file_size, rows); invalidated by our own appends.
        self._cache: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}

    # ------------------------------------------------------------------ paths

    @staticmethod
    def _check_role(role_uuid: str) -> str:
        if not role_uuid or "/" in role_uuid or "\\" in role_uuid or role_uuid in {".", ".."}:
            raise ValueError("invalid role UUID")
        return role_uuid

    def path(self, role_uuid: str) -> Path:
        return self.root / f"{self._check_role(role_uuid)}.jsonl"

    # ------------------------------------------------------------------ write

    def record_turn(
        self,
        role_uuid: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        *,
        at: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Extract and append this turn's dated sentences. Returns the new rows."""
        # ``at`` may arrive as a datetime or as the ISO string a caller stored;
        # parsing here keeps both paths resolving against the same moment.
        moment = parse_moment(at) or datetime.now()
        rows = self.rows_for_turn(
            role_uuid, conversation_id, user_content, assistant_content, at=moment
        )
        if rows:
            self._append(role_uuid, rows)
        return rows

    def rows_for_turn(
        self,
        role_uuid: str,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        *,
        at: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """The rows this turn *would* produce (pure; no I/O)."""
        moment = parse_moment(at) or datetime.now()
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for role, body in (("user", user_content), ("assistant", assistant_content)):
            for sentence in sentences(body):
                hits = temporal_hits(sentence, moment)
                if not hits:
                    continue
                days = sorted({day.isoformat() for hit in hits for day in hit["days"]})
                label = days[0] if days else moment.date().isoformat()
                row_id = _row_id(role_uuid, conversation_id, label, sentence)
                if row_id in seen:
                    continue
                seen.add(row_id)
                rows.append({
                    "id": row_id,
                    "conversation_id": str(conversation_id or ""),
                    "speaker": role,
                    "sentence": sentence,
                    "days": days,
                    "kinds": sorted({hit["kind"] for hit in hits}),
                    "at": moment.replace(microsecond=0).isoformat(),
                    "created_at": utc_timestamp(),
                })
                if len(rows) >= _MAX_ROWS_PER_TURN:
                    return rows
        return rows

    def _append(self, role_uuid: str, rows: Sequence[Dict[str, Any]]) -> None:
        path = self.path(role_uuid)
        path.parent.mkdir(parents=True, exist_ok=True)
        known = {str(row.get("id")) for row in self.rows(role_uuid)}
        fresh = [row for row in rows if str(row.get("id")) not in known]
        if not fresh:
            return
        with path.open("a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        self._cache.pop(role_uuid, None)

    # ------------------------------------------------------------------- read

    def rows(self, role_uuid: str, *, use_cache: bool = True) -> List[Dict[str, Any]]:
        path = self.path(role_uuid)
        if not path.exists():
            self._cache.pop(role_uuid, None)
            return []
        try:
            size = path.stat().st_size
        except OSError:
            return []
        if use_cache:
            cached = self._cache.get(role_uuid)
            if cached is not None and cached[0] == size:
                return cached[1]
        entries: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict) and row.get("sentence"):
                        entries.append(row)
        except OSError:
            return []
        entries = entries[-self.max_entries :]
        if use_cache:
            self._cache[role_uuid] = (size, entries)
        return entries

    def search(
        self,
        role_uuid: str,
        query: str,
        *,
        reference: Optional[datetime] = None,
        limit: int = 3,
        lexical_floor: float = 0.30,
    ) -> Dict[str, Any]:
        """Rows the query can address, newest first within each access mode.

        Day-addressed rows come first: an explicit day in the query is a hard
        constraint, so a lexical match on another day must not displace it.
        """
        cap = max(0, int(limit))
        empty = {"status": "empty", "mode": "none", "days": [], "items": [], "candidates": 0}
        if cap <= 0:
            return dict(empty, status="disabled")
        rows = self.rows(role_uuid)
        if not rows:
            return empty
        days = set(query_days(query, reference))
        by_day = [row for row in rows if days & set(row.get("days") or ())]
        by_day.sort(key=lambda row: (str(row.get("at") or ""), str(row.get("id") or "")), reverse=True)
        picked: List[Dict[str, Any]] = by_day[:cap]
        mode = "day" if picked else "none"
        if len(picked) < cap and is_time_question(query):
            taken = {str(row.get("id")) for row in picked}
            scored = [
                (score_text(query, str(row.get("sentence") or "")), row)
                for row in rows
                if str(row.get("id")) not in taken
            ]
            scored = [item for item in scored if item[0] >= lexical_floor]
            scored.sort(key=lambda item: (item[0], str(item[1].get("at") or "")), reverse=True)
            for score, row in scored[: cap - len(picked)]:
                picked.append(dict(row, score=round(float(score), 4)))
            mode = "day+content" if by_day else "content"
        return {
            "status": "ok" if picked else "empty",
            "mode": mode,
            "days": sorted(days),
            "items": picked,
            "candidates": len(by_day),
        }

    # ------------------------------------------------------------- lifecycle

    def purge(self, role_uuid: str) -> int:
        path = self.path(role_uuid)
        self._cache.pop(role_uuid, None)
        if not path.exists():
            return 0
        try:
            path.unlink()
            return 1
        except OSError:
            return 0

    def stats(self, role_uuid: str) -> Dict[str, Any]:
        rows = self.rows(role_uuid)
        return {
            "rows": len(rows),
            "with_day": sum(1 for row in rows if row.get("days")),
            "conversations": len({str(row.get("conversation_id")) for row in rows}),
            "path": str(self.path(role_uuid)).replace("\\", "/"),
        }


def day_label(days: Iterable[str]) -> str:
    """How a row names its day(s) in the injected text.

    A month-level expression stores every day of that month; labelling that with
    its first day would claim "1 July" for what the speaker said as "in July".
    """
    ordered = sorted({str(day) for day in days if day})
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    first, last = ordered[0], ordered[-1]
    if first[:7] == last[:7] and first.endswith("-01") and len(ordered) >= 28:
        return first[:7]
    return f"{first}…{last}"


def render_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """One calendar row as a recall item, with its day in the text."""
    label = day_label(row.get("days") or ()) or str(row.get("at") or "")[:10]
    sentence = str(row.get("sentence") or "").strip()
    content = f"{label} · {sentence}" if label else sentence
    return {
        "id": str(row.get("id") or ""),
        "content": content,
        "layer": "Calendar",
        "match_type": "calendar",
        "calendar_day": label,
        "created_at": str(row.get("at") or row.get("created_at") or ""),
        "token_estimate": len(content),
        "final_score": float(row.get("score") or 0.0),
    }
