"""Interactive query script for the Fatawa RAG system.

Usage
-----
    python run_query.py                  # interactive REPL
    python run_query.py "آپ کا سوال"     # single query from CLI arg
"""

from __future__ import annotations

import io
import sys
import textwrap

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from src.pipeline import query  # noqa: E402  (import after dotenv)


# ── Display helpers ───────────────────────────────────────────────────────────

_SEP  = "─" * 60
_SEP2 = "═" * 60


def _print_answer(answer: str) -> None:
    print(f"\n{_SEP2}")
    print("  جواب (Answer)")
    print(_SEP2)
    # Wrap long lines for terminal readability
    for line in answer.splitlines():
        print(textwrap.fill(line, width=80) if line.strip() else "")
    print(_SEP2)


def _print_sources(sources: list[dict], n: int = 3) -> None:
    top = sources[:n]
    if not top:
        print("\n  (کوئی ماخذ دستیاب نہیں)")
        return

    print(f"\n  ماخذ (Top {len(top)} Sources)")
    print(_SEP)
    for i, meta in enumerate(top, 1):
        question    = (meta.get("question") or "").strip()
        category    = meta.get("category", "نامعلوم")
        source_name = (
            meta.get("source_name")
            or (meta.get("folder", "") or meta.get("source") or meta.get("source_file", ""))
               .replace("-ExtractedData-Output", "").strip()
            or "نامعلوم"
        )
        fatwa_no    = meta.get("fatwa_no", "")

        # Truncate long questions for display
        if len(question) > 120:
            question = question[:120].rsplit(" ", 1)[0] + "…"

        maslak = meta.get("maslak", "")
        print(f"  [{i}] ڈیٹا ماخذ : {source_name}")
        if maslak:
            print(f"       مسلک    : {maslak}")
        print(f"       زمرہ    : {category}")
        if fatwa_no:
            print(f"       فتویٰ نمبر: {fatwa_no}")
        if question:
            print(f"       سوال    : {question}")
        print()


def _print_timings(timings: dict) -> None:
    total = timings.get("total_ms", 0)
    ret   = timings.get("retrieve_ms", 0)
    gen   = timings.get("generate_ms", 0)
    print(_SEP)
    print(f"  ⏱  retrieve {ret:.0f}ms  |  generate {gen:.0f}ms  |  total {total:.0f}ms")
    print(_SEP)


# ── Core run function ─────────────────────────────────────────────────────────

def run(user_query: str) -> None:
    print(f"\n{_SEP}")
    print(f"  سوال: {user_query.strip()}")
    print(_SEP)
    print("  جواب تیار ہو رہا ہے…\n")

    result = query(user_query)

    _print_answer(result["answer"])
    _print_sources(result["sources"], n=3)
    _print_timings(result["timings"])


# ── Entry points ──────────────────────────────────────────────────────────────

def main() -> None:
    # Single query from CLI argument
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
        return

    # Interactive REPL
    print(_SEP2)
    print("  اسلامی فتاویٰ RAG سسٹم")
    print("  Islamic Fatawa RAG System")
    print(f"  (type 'exit' or press Ctrl-C to quit)")
    print(_SEP2)

    while True:
        try:
            user_input = input("\nسوال درج کریں: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nخدا حافظ۔")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "خروج"}:
            print("خدا حافظ۔")
            break

        try:
            run(user_input)
        except Exception as exc:
            print(f"\n  خرابی پیش آئی: {exc}")


if __name__ == "__main__":
    main()
