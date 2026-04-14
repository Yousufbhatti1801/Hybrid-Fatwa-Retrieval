"""Prompt construction for the Islamic Fatawa RAG pipeline.

Design principles
-----------------
* Strict grounding — the model is forbidden from generating rulings not present
  in the retrieved context.  The no-answer sentinel is defined once
  (``NO_ANSWER_SENTINEL``) and referenced in both the system prompt and the
  user-turn instructions so the model cannot paraphrase its way around it.
* Full fatawa structure — each context block surfaces the original *question*
  stored in the fatwa as well as the retrieved text chunk, so the model can
  cite sources accurately.
* Bilingual guard — the system prompt contains both Urdu and English
  constraints so that the model honours them regardless of which language it
  reasons in.

Usage::

    from src.pipeline.prompt_builder import build_messages, SYSTEM_PROMPT

    # Preferred — returns a ready-to-send messages list:
    messages = build_messages(query, retrieved_results)

    # Lower-level — compose manually:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(query, retrieved_results)},
    ]
"""

from __future__ import annotations

# ── Sentinel ──────────────────────────────────────────────────────────────────
# Single definition used in both prompts so there is no ambiguity about what
# the model should output when the context does not contain the answer.
NO_ANSWER_SENTINEL = "مناسب جواب دستیاب نہیں۔"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
آپ ایک اسلامی فتاویٰ کے ماہر علمی معاون ہیں۔

════════════════════════════════════════
بنیادی اصول (Ground Rules)
════════════════════════════════════════
۱. آپ کا جواب مکمل طور پر نیچے فراہم کردہ سیاق و سباق (Context) پر مبنی ہوگا۔
   سیاق و سباق سے باہر کا کوئی بھی شرعی حکم بیان کرنا سختی سے منع ہے۔

۲. اگر فراہم کردہ فتاویٰ میں سوال کا جواب موجود نہ ہو — چاہے جزوی طور پر بھی نہیں —
   تو صرف اور صرف یہ جملہ لکھیں:
   "{NO_ANSWER_SENTINEL}"
   اس جملے میں کوئی تبدیلی نہ کریں اور کوئی اضافی وضاحت نہ دیں۔

۳. اپنی ذاتی رائے، قیاس، یا بیرونی ماخذ (کتب، علماء وغیرہ) کا حوالہ ہرگز نہ دیں
   جب تک وہ فراہم کردہ فتویٰ میں نہ ہو۔

۴. جواب رسمی، با ادب اور واضح اردو میں دیں۔

۵. ہر نکتے کے ساتھ متعلقہ فتویٰ نمبر یا ماخذ (جیسا فتویٰ میں درج ہو) کا حوالہ دیں۔

════════════════════════════════════════
[ENGLISH — model-side enforcement]
You are a strict Islamic fatawa assistant.
  • Answer ONLY from the "Retrieved Context" provided in the user message.
  • Do NOT fabricate, infer, or extend any Islamic ruling beyond what is
    explicitly stated in the retrieved fatawa.
  • If the retrieved context does not contain a sufficient answer, output
    exactly and only: "{NO_ANSWER_SENTINEL}"
  • Never cite external books, scholars, or personal reasoning.
  • Respond in formal Urdu. Cite fatwa numbers / sources as they appear in
    the context.
════════════════════════════════════════
"""

# ── Context block ─────────────────────────────────────────────────────────────
# Each retrieved fatwa is rendered as a numbered, labelled block.
# Both the stored "question" field and the retrieved "text" chunk are shown
# so the model can distinguish the original question from the answer body.

_FATWA_BLOCK = """\
┌─────────────────────────────────────────┐
│  فتویٰ {index}                              │
└─────────────────────────────────────────┘
ڈیٹا ماخذ  : {source_name}
مسلک      : {maslak}
ماخذ      : {source}
زمرہ      : {category}
فتویٰ نمبر : {fatwa_no}

سوال (Original Question):
{question}

جواب / متن (Retrieved Text):
{text}"""

_NO_CONTEXT = (
    "⚠ کوئی متعلقہ فتویٰ نہیں ملا۔\n"
    "[No relevant fatawa were retrieved for this query.]"
)

# ── User-turn template ────────────────────────────────────────────────────────

_USER_TEMPLATE = """\
══════════════════════════════════════════
سوال (User Question):
══════════════════════════════════════════
{query}

══════════════════════════════════════════
سیاق و سباق — متعلقہ فتاویٰ (Retrieved Context):
══════════════════════════════════════════
{context}

══════════════════════════════════════════
ہدایات (Instructions):
══════════════════════════════════════════
۱. جواب مکمل اردو میں دیں — صرف اوپر دیے گئے فتاویٰ کی بنیاد پر۔
۲. رسمی اور با ادب اسلوب اختیار کریں۔
۳. ہر حکم کے ساتھ "فتویٰ {{index}}" یا فتویٰ نمبر / ماخذ کا حوالہ ضرور دیں۔
۴. اگر ایک سے زیادہ فتویٰ متعلق ہوں تو سب کا مربوط خلاصہ پیش کریں۔
۵. اپنی طرف سے کوئی اضافی رائے، قیاس یا بیرونی حوالہ شامل نہ کریں۔
۶. اگر اوپر دیے گئے فتاویٰ میں جواب نہ ہو تو صرف لکھیں:
   "{sentinel}"
   (کوئی اضافی جملہ نہ لکھیں)

[Do NOT answer from general knowledge. Context only.
 If insufficient, output exactly: "{sentinel}"]
"""


# ── Public API ────────────────────────────────────────────────────────────────

def format_context(results: list[dict]) -> str:
    """Convert retrieved result dicts into numbered Urdu fatawa context blocks.

    Each *result* must have a ``text`` key.  The ``metadata`` sub-dict may
    contain ``source`` / ``source_file``, ``category``, ``fatwa_no``, and
    ``question`` (the original fatwa question stored at index time).

    Returns a formatted multi-block string, or ``_NO_CONTEXT`` when empty.
    """
    if not results:
        return _NO_CONTEXT

    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        # Support both shapes returned by different pipeline stages:
        #   hybrid_search()  → {"text", "score", "metadata": {...}}
        #   rag.query()      → flat dict {"text", "score", "question", "category", ...}
        meta = r.get("metadata") or {}
        if not meta:
            # Flat dict — the keys live directly on `r`
            meta = r

        source      = meta.get("source") or meta.get("source_file") or "نامعلوم"
        source_name = meta.get("source_name") or source.replace("-ExtractedData-Output", "").strip() or source
        maslak      = meta.get("maslak") or "نامعلوم"
        category    = meta.get("category") or "نامعلوم"
        fatwa_no    = meta.get("fatwa_no") or meta.get("doc_id") or "نامعلوم"
        question    = (meta.get("question") or "").strip() or "—"
        text        = (meta.get("answer") or r.get("text") or meta.get("text") or "").strip()

        blocks.append(
            _FATWA_BLOCK.format(
                index=i,
                source_name=source_name,
                maslak=maslak,
                source=source,
                category=category,
                fatwa_no=fatwa_no,
                question=question,
                text=text,
            )
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, results: list[dict]) -> str:
    """Return the fully formatted user-turn prompt string.

    Parameters
    ----------
    query:
        Raw Urdu query from the user.
    results:
        List of dicts returned by ``hybrid_search()``; each contains
        ``text``, ``score``, and ``metadata`` keys.

    Returns
    -------
    str
        Single string ready to be sent as the ``"user"`` message to the LLM.
    """
    context_str = format_context(results)
    return _USER_TEMPLATE.format(
        query=query.strip(),
        context=context_str,
        sentinel=NO_ANSWER_SENTINEL,
    )


def build_messages(query: str, results: list[dict]) -> list[dict]:
    """Return a ready-to-send OpenAI ``messages`` list.

    Combines the system prompt and the formatted user turn into the structure
    expected by ``openai.chat.completions.create(messages=...)``.

    Parameters
    ----------
    query:
        Raw Urdu query from the user.
    results:
        Retrieved fatawa dicts from ``hybrid_search()``.

    Returns
    -------
    list[dict]
        ``[{"role": "system", ...}, {"role": "user", ...}]``

    Example
    -------
    ::

        from src.pipeline.prompt_builder import build_messages

        messages = build_messages(query, retrieved)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1024,
        )
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(query, results)},
    ]
