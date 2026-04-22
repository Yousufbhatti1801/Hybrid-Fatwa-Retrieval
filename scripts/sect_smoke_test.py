"""Quick static smoke test for the sect-aware retrieval plumbing.

This test does NOT load the BM25 corpus or hit any network — it only
exercises the pure-Python parts that are cheap to test:

    * ``detect_sect_in_query`` — keyword-based sect detection
    * ``_sect_and_source_for_path`` — folder → (sect, source) mapping
    * ``SECT_TO_SOURCES`` / ``SOURCE_TO_SECT`` — taxonomy consistency

Run: ``python scripts/sect_smoke_test.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.islam360.retrieve import detect_sect_in_query  # noqa: E402
from src.islam360.url_index import (  # noqa: E402
    ALL_SECTS,
    SECT_AHLE_HADITH,
    SECT_BARELVI,
    SECT_DEOBANDI,
    SECT_TO_SOURCES,
    SOURCE_TO_SECT,
    _sect_and_source_for_path,
)


def expect(name: str, got, want) -> bool:
    ok = got == want
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}: got={got!r}  want={want!r}")
    return ok


def main() -> int:
    passed = 0
    total = 0

    print("-- taxonomy consistency --")
    for src, sect in SOURCE_TO_SECT.items():
        assert src in SECT_TO_SOURCES[sect], (src, sect, SECT_TO_SOURCES)
    for sect, sources in SECT_TO_SOURCES.items():
        for s in sources:
            assert SOURCE_TO_SECT[s] == sect, (s, sect, SOURCE_TO_SECT[s])
    print(f"  SECT_TO_SOURCES:  {dict(SECT_TO_SOURCES)}")
    print(f"  SOURCE_TO_SECT:   {SOURCE_TO_SECT}")

    print("\n-- path → (sect, source) mapping --")
    cases = [
        (Path("data/Islam-360-Fatwa-data/Banuri-ExtractedData-Output/NAMAZ/a.csv"),
         (SECT_DEOBANDI, "banuri")),
        (Path("data/Islam-360-Fatwa-data/urdufatwa-ExtractedData-Output/ZAKAT/z.csv"),
         (SECT_BARELVI, "urdu_fatwa")),
        (Path("data/Islam-360-Fatwa-data/IslamQA-ExtractedData-Output/HAJJ/h.csv"),
         (SECT_AHLE_HADITH, "ahle_hadith_1")),
        (Path("data/Islam-360-Fatwa-data/fatwaqa-ExtractedData-Output/FAST/f.csv"),
         (SECT_AHLE_HADITH, "ahle_hadith_2")),
        (Path("data/Islam-360-Fatwa-data/Unknown/foo.csv"), ("", "")),
    ]
    for p, want in cases:
        total += 1
        got = _sect_and_source_for_path(p)
        if expect(str(p), got, want):
            passed += 1

    print("\n-- sect detection from query --")
    query_cases = [
        ("deobandi fatwa de do",                             SECT_DEOBANDI),
        ("barelvi mufti kya kehte hain",                     SECT_BARELVI),
        ("ahle hadees ka moaqqif",                           SECT_AHLE_HADITH),
        ("ahl-e-hadith kay nazdeek",                         SECT_AHLE_HADITH),
        ("banuri wala fatwa",                                SECT_DEOBANDI),
        ("urdufatwa par kya likha hai",                      SECT_BARELVI),
        ("islamqa say jawab lao",                            SECT_AHLE_HADITH),
        ("salafi ka moaqqif",                                SECT_AHLE_HADITH),
        # Urdu-script variants
        ("دیوبندی مفتی کا فتویٰ",                              SECT_DEOBANDI),
        ("بریلوی عالم کے مطابق",                             SECT_BARELVI),
        ("اہل حدیث کیا کہتے ہیں",                            SECT_AHLE_HADITH),
        # Sect-neutral → None
        ("namaz ka tareeqa",                                 None),
        ("zakat kis par farz hai",                           None),
        ("bandar ko hath lagane se kapre napak ho jate hai", None),
        # Tricky false-positive candidates (must NOT trigger)
        ("bandar ko chuney se wudu",                         None),  # "bandar" ≠ "banuri"
        ("islami qawaneen",                                  None),  # "islam" substring only
    ]
    for q, want in query_cases:
        total += 1
        got = detect_sect_in_query(q)
        if expect(repr(q), got, want):
            passed += 1

    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
