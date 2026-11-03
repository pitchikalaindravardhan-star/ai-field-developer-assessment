"""Incremental PII redaction for text that arrives in pieces.

The hard part is that a value can be split across deltas: "john.d", "oe@exa",
"mple.com". Regex on each delta misses it, and regex on the full response means
buffering the whole thing. So this holds back only the tail of the text that
could still turn into a match and releases everything before it.

The held tail is the trailing run of characters that can belong to an email or a
number. It stays small because the run ends at the first character outside those
sets, and it has a hard cap because the longest possible match is bounded.

Cost per feed is O(chunk + tail), so a stream of any length stays linear and
memory stays under one chunk plus MAX_BUFFER characters (about 570).
"""

import re

REPLACEMENT = "[REDACTED]"

_EMAIL_LOCAL = r"[\w.%+-]{1,64}"
_EMAIL_DOMAIN = r"(?:[^\W_](?:[\w-]{0,61}[^\W_])?\.){1,6}[^\W\d_]{2,24}"

# One pattern for everything so the text is scanned once. A number that isn't a
# card or SSN costs a single callback.
_PII = re.compile(
    rf"(?P<email>{_EMAIL_LOCAL}@{_EMAIL_DOMAIN})"
    r"|(?P<number>(?<![0-9])[0-9](?:[ -]?[0-9]){8,47})"
)
_SSN = re.compile(r"(?<![0-9])[0-9]{3}[ -][0-9]{2}[ -][0-9]{4}(?![0-9])")
_DIGIT_GROUP = re.compile(r"[0-9]+")

# Longest possible match of each alternative, from the bounds in the patterns.
_MAX_EMAIL = 64 + 1 + 6 * 64 + 24
_MAX_NUMBER = 48 + 47
MAX_HOLD = max(_MAX_EMAIL, _MAX_NUMBER)
# A match cut by the hold cap is kept back whole, so it can add its own length.
MAX_BUFFER = MAX_HOLD + _MAX_NUMBER

_EMAIL_PUNCTUATION = frozenset("._%+-@")
_DIGITS = frozenset("0123456789")
_NUMBER_SEPARATORS = frozenset(" -")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if position % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _card_spans(text: str) -> list[tuple[int, int]]:
    """Finds Luhn-valid cards in a run of digit groups.

    The run may hold more than the card ("order 12 4111 1111 1111 1111 x2"), so
    every contiguous set of groups gets tried, not just the run as a whole. About
    one in ten random windows passes Luhn, so a wrong window can overlap the real
    card. Overlapping hits are merged, which redacts a little too much at worst
    and never leaves part of a card behind.
    """
    groups = list(_DIGIT_GROUP.finditer(text))
    spans = []
    for first in range(len(groups)):
        digits = ""
        for last in range(first, len(groups)):
            digits += groups[last].group()
            if len(digits) > 19:
                break
            if len(digits) >= 13 and _luhn_valid(digits):
                spans.append((groups[first].start(), groups[last].end()))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _redact_cards(text: str) -> str:
    pieces, cursor = [], 0
    for start, end in _card_spans(text):
        pieces += [text[cursor:start], REPLACEMENT]
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_number(text: str) -> str:
    # SSNs first: a card is never grouped 3-2-4, but digits from an SSN sitting
    # next to a card can form a false card window with it.
    pieces, cursor = [], 0
    for match in _SSN.finditer(text):
        pieces += [_redact_cards(text[cursor : match.start()]), REPLACEMENT]
        cursor = match.end()
    pieces.append(_redact_cards(text[cursor:]))
    return "".join(pieces)


def _replace(match: re.Match) -> str:
    if match.lastgroup == "email":
        return REPLACEMENT
    return _redact_number(match.group())


def _hold_start(text: str) -> int:
    """Index where the part of `text` that might still grow into a match begins."""
    length = len(text)
    floor = max(0, length - MAX_HOLD)

    email_start = length
    while email_start > floor:
        char = text[email_start - 1]
        if not (char.isalnum() or char in _EMAIL_PUNCTUATION):
            break
        email_start -= 1

    number_start = length
    index = length
    while index > floor:
        char = text[index - 1]
        if char in _DIGITS:
            number_start = index - 1
        elif char not in _NUMBER_SEPARATORS:
            break
        index -= 1

    return min(email_start, number_start)


class StreamRedactor:
    """Feed it deltas in order. Each call returns the text that is safe to send."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending = ""

    @property
    def pending_chars(self) -> int:
        return len(self._pending)

    def feed(self, delta: str) -> str:
        text = self._pending + delta if self._pending else delta
        safe_end = _hold_start(text)
        output, consumed = self._scan(text, safe_end)
        self._pending = text[consumed:]
        return output

    def flush(self, truncated: bool = False) -> str:
        """Call once at the end of the stream.

        With truncated=True the text was cut off (length limit, upstream failure),
        so the tail might be half a value that no pattern can recognize, like
        "john@gmail" or "4111 1111 1111". That gets redacted rather than sent.
        """
        text, self._pending = self._pending, ""
        output, _ = self._scan(text, len(text))
        tail = output.rsplit(REPLACEMENT, 1)[-1]
        if truncated and ("@" in tail or any(c in _DIGITS for c in tail)):
            return output[: len(output) - len(tail)] + REPLACEMENT
        return output

    @staticmethod
    def _scan(text: str, safe_end: int) -> tuple[str, int]:
        """Redacts text[:safe_end]. Returns (output, characters consumed)."""
        pieces = []
        cursor = 0
        # Cheap pre-check: most deltas are plain words, which skip the regex.
        if "@" in text or any(c in _DIGITS for c in text):
            for match in _PII.finditer(text):
                if match.start() >= safe_end:
                    break
                if match.end() > safe_end:
                    # Only reachable when the hold cap cut through a long run.
                    # Keep the whole match back so none of it is released.
                    safe_end = match.start()
                    break
                pieces.append(text[cursor : match.start()])
                pieces.append(_replace(match))
                cursor = match.end()
        pieces.append(text[cursor:safe_end])
        return "".join(pieces), safe_end
