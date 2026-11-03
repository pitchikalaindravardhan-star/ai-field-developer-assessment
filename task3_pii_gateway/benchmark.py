"""Measures what the gateway adds to a stream. Run: python -m task3_pii_gateway.benchmark"""

import time
import tracemalloc

from .redactor import StreamRedactor

SENTENCE = "The quarterly report for account 8842 was sent to jane.doe@example.com yesterday. "


def main() -> None:
    deltas = [SENTENCE[i : i + 4] for i in range(0, len(SENTENCE), 4)] * 20_000
    characters = sum(map(len, deltas))

    redactor = StreamRedactor()
    tracemalloc.start()
    worst_ns = 0
    started = time.perf_counter_ns()
    for delta in deltas:
        before = time.perf_counter_ns()
        redactor.feed(delta)
        worst_ns = max(worst_ns, time.perf_counter_ns() - before)
    elapsed_ns = time.perf_counter_ns() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"deltas:          {len(deltas):,}")
    print(f"text processed:  {characters / 1_000_000:.1f} MB")
    print(f"mean per delta:  {elapsed_ns / len(deltas) / 1000:.1f} us (with tracemalloc on)")
    print(f"worst delta:     {worst_ns / 1000:.0f} us")
    print(f"peak memory:     {peak / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
