import random
import time

import pytest

from task3_pii_gateway.redactor import MAX_BUFFER, REPLACEMENT, StreamRedactor


def redact_in_chunks(text, sizes):
    redactor = StreamRedactor()
    output, position = [], 0
    while position < len(text):
        size = next(sizes)
        output.append(redactor.feed(text[position : position + size]))
        position += size
    output.append(redactor.flush())
    return "".join(output)


def redact_whole(text):
    redactor = StreamRedactor()
    return redactor.feed(text) + redactor.flush()


def fixed(size):
    while True:
        yield size


def randomized(seed):
    rng = random.Random(seed)
    while True:
        yield rng.randint(1, 12)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail jane.doe@example.com now", f"mail {REPLACEMENT} now"),
        ("JANE+tag@Sub.Example.CO.UK.", f"{REPLACEMENT}."),
        ("(a_b-c%d@x-y.io)", f"({REPLACEMENT})"),
        ("josé@example.com", REPLACEMENT),
        ("ssn 123-45-6789.", f"ssn {REPLACEMENT}."),
        ("ssn 123 45 6789 ok", f"ssn {REPLACEMENT} ok"),
        ("visa 4111111111111111", f"visa {REPLACEMENT}"),
        ("visa 4111 1111 1111 1111!", f"visa {REPLACEMENT}!"),
        ("visa 4111-1111-1111-1111 ok", f"visa {REPLACEMENT} ok"),
        ("amex 3782 822463 10005.", f"amex {REPLACEMENT}."),
        ("two a@b.co and c@d.io", f"two {REPLACEMENT} and {REPLACEMENT}"),
        ("id 12 4111 1111 1111 1111 x", f"id 12 {REPLACEMENT} x"),
        ("4111 1111 1111 1111 12 ok", f"{REPLACEMENT} 12 ok"),
        ("123-45-6789 4111 1111 1111 1111", f"{REPLACEMENT} {REPLACEMENT}"),
    ],
)
def test_detects_pii(text, expected):
    assert redact_whole(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "plain sentence with no secrets",
        "released 2024-01-15 at 10:30",
        "call 972-555-0199 or 1-800-555-0199",
        "1234567890123456 fails luhn",
        "version 1.2.3 of a@b",
        "tweet @someone about it",
        "12-3456-789 wrong ssn shape",
        "1234-56-7890 wrong ssn shape",
    ],
)
def test_leaves_normal_text_alone(text):
    assert redact_whole(text) == text


SAMPLE = (
    "Reach jane.doe@example.com or bob@sub.example.co.uk, SSN 123-45-6789 and "
    "078 05 1120. Card 4111 1111 1111 1111, also 5500-0000-0000-0004, and "
    "378282246310005. On 2024-01-15 order 8842 shipped to room 12, thanks."
)


def test_same_result_for_every_split_position():
    expected = redact_whole(SAMPLE)
    for split in range(len(SAMPLE)):
        redactor = StreamRedactor()
        result = redactor.feed(SAMPLE[:split]) + redactor.feed(SAMPLE[split:]) + redactor.flush()
        assert result == expected, split


def test_same_result_one_character_at_a_time():
    assert redact_in_chunks(SAMPLE, fixed(1)) == redact_whole(SAMPLE)


@pytest.mark.parametrize("seed", range(200))
def test_same_result_for_random_chunking(seed):
    assert redact_in_chunks(SAMPLE, randomized(seed)) == redact_whole(SAMPLE)


def test_no_fragment_of_a_secret_is_ever_released():
    redactor = StreamRedactor()
    released = ""
    for char in "contact jane.doe@example.com, thanks":
        released += redactor.feed(char)
        assert "jane" not in released
        assert "example" not in released
    assert redactor.flush() == "thanks" or REPLACEMENT in released


def test_plain_text_is_released_at_the_next_word_boundary():
    redactor = StreamRedactor()
    assert redactor.feed("Hello") == ""
    assert redactor.feed(" there") == "Hello "
    assert redactor.feed(" friend") == "there "


def test_number_is_held_until_it_can_be_judged():
    redactor = StreamRedactor()
    assert redactor.feed("id 4111 1111") == "id "
    assert redactor.feed(" 1111 1111") == ""
    assert redactor.feed(" ok") == f"{REPLACEMENT} "
    assert redactor.flush() == "ok"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mail john@gmail", f"mail {REPLACEMENT}"),
        ("card 4111 1111 1111", f"card {REPLACEMENT}"),
        ("the end of a sentence", "the end of a sentence"),
    ],
)
def test_truncated_stream_does_not_release_half_a_value(text, expected):
    redactor = StreamRedactor()
    assert redactor.feed(text) + redactor.flush(truncated=True) == expected


def test_memory_stays_bounded_without_any_break_in_the_text():
    redactor = StreamRedactor()
    for filler in ("a", "1", "1 ", "a.", "a@"):
        for _ in range(20_000):
            redactor.feed(filler)
            assert redactor.pending_chars <= MAX_BUFFER + len(filler)
        redactor.flush()


def test_long_digit_run_does_not_stall_the_stream():
    redactor = StreamRedactor()
    released = sum(len(redactor.feed("7")) for _ in range(5_000))
    assert released > 4_000


def test_secret_after_long_filler_is_still_caught():
    filler = "x" * 5000 + " " + "1 " * 500
    text = filler + "jane@example.com " + "4111 1111 1111 1111 done"
    result = redact_in_chunks(text, fixed(3))
    assert "jane@example.com" not in result
    assert "4111" not in result


def test_throughput_is_linear_in_stream_length():
    def elapsed(deltas):
        redactor = StreamRedactor()
        start = time.perf_counter()
        for _ in range(deltas):
            redactor.feed("word ")
        return time.perf_counter() - start

    small, large = elapsed(20_000), elapsed(200_000)
    assert large < small * 25  # ten times the text, allowing for noise
