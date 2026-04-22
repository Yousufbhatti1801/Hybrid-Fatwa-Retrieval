/* ── State ─────────────────────────────────────────────────────────────── */
const state = {
  selectedCat: 'ALL',
  history: [],
  isLoading: false,
  lastSources: [],
  mode: 'hybrid',     // 'hybrid' | 'pageindex' only
};

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const welcomeState   = document.getElementById('welcomeState');
const messages       = document.getElementById('messages');
const questionInput  = document.getElementById('questionInput');
const sendBtn        = document.getElementById('sendBtn');
const catList        = document.getElementById('catList');
const historyList    = document.getElementById('historyList');
const guardrailsToggle = document.getElementById('guardrailsToggle');
const validateToggle   = document.getElementById('validateToggle');
const topKInput        = document.getElementById('topKInput');
const modalOverlay     = document.getElementById('modalOverlay');
const modal            = document.getElementById('modal');
const modalClose       = document.getElementById('modalClose');
const modalBody        = document.getElementById('modalBody');

/* ── Category selection ────────────────────────────────────────────────── */
catList.addEventListener('click', e => {
  const btn = e.target.closest('.cat-btn');
  if (!btn) return;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.selectedCat = btn.dataset.cat;
});

/* ── Retrieval mode toggle (Hybrid / PageIndex) ────────────────────────── */
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.mode = btn.dataset.mode || 'hybrid';
  });
});

/* ── Suggestion chips ──────────────────────────────────────────────────── */
document.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  questionInput.value = chip.dataset.q;
  questionInput.dispatchEvent(new Event('input'));
  sendQuestion();
});

/* ── Auto-resize textarea ──────────────────────────────────────────────── */
questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + 'px';
});

/* ── Send on Enter (Shift+Enter = newline) ─────────────────────────────── */
questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
});

sendBtn.addEventListener('click', sendQuestion);

/* ── Modal ──────────────────────────────────────────────────────────────── */
modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

function openModal(sources) {
  modalBody.innerHTML = sources.map((s, i) => buildSourceCard(s, i + 1)).join('');
  modalOverlay.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeModal() {
  modalOverlay.classList.remove('open');
  document.body.style.overflow = '';
}

/* ── Main send function ─────────────────────────────────────────────────── */
async function sendQuestion() {
  const q = questionInput.value.trim();
  if (!q || state.isLoading) return;

  // Hide welcome
  if (welcomeState) welcomeState.style.display = 'none';

  // Append user bubble
  appendUserBubble(q);

  // Clear input
  questionInput.value = '';
  questionInput.style.height = 'auto';

  // Branch on retrieval mode
  if (state.mode === 'pageindex') {
    return doPISearch(q);
  }

  // Show typing indicator
  const typingId = appendTyping();
  setLoading(true);

  try {
    const payload = {
      question:    q,
      category:    state.selectedCat,
      top_k:       parseInt(topKInput.value) || 5,
      guardrails:  guardrailsToggle.checked,
      validate:    validateToggle.checked,
    };

    // Prefer multi-school endpoint so user gets separate answers per sect.
    let resp = await fetch('/api/query-all-schools', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    removeTyping(typingId);

    if (resp.ok) {
      const data = await resp.json();
      const results = Array.isArray(data.results) ? data.results : [];

      if (results.length) {
        renderHybridResults(data);
        addHistory(q);
      } else {
        // Fallback to original single-answer endpoint
        resp = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({ error: resp.statusText }));
          appendErrorBubble(err.error || 'Server error');
        } else {
          const single = await resp.json();
          appendBotBubble(single);
          addHistory(q);
        }
      }
    } else {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      appendErrorBubble(err.error || 'Server error');
    }
  } catch (err) {
    removeTyping(typingId);
    appendErrorBubble('شبکہ خطا — Network error: ' + err.message);
  } finally {
    setLoading(false);
    scrollBottom();
  }
}

/* ── Bubble builders ────────────────────────────────────────────────────── */
function appendUserBubble(text) {
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML = `<div class="bubble">${escHtml(text)}</div>`;
  messages.appendChild(el);
  scrollBottom();
}

function appendTyping() {
  const id = 'typing-' + Date.now();
  const el = document.createElement('div');
  el.id = id;
  el.className = 'msg msg-bot';
  el.innerHTML = `
    <div class="card">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendBotBubble(data) {
  const el = document.createElement('div');
  el.className = 'msg msg-bot';

  const badge     = buildStatusBadge(data);
  const sectBadge = data.maslak ? `<span class="badge badge-ok">${escHtml(data.maslak)}</span>` : '';
  const guardHtml = buildGuardHits(data.guard_hits || []);
  const valHtml   = buildValBar(data.validation);
  const footHtml  = buildFooter(data);
  const sectLine  = data.maslak ? `<p class="answer-text"><strong>مسلک:</strong> ${escHtml(data.maslak)}</p>` : '';
  // If the preprocessor translated a non-Urdu query, show how it was interpreted.
  const showInterp =
    data.original_question &&
    data.urdu_question &&
    data.original_question.trim() !== data.urdu_question.trim();
  const interpLine = showInterp
    ? `<p class="answer-interp"><em>تلاش بطور:</em> ${escHtml(data.urdu_question)}</p>`
    : '';
  const bodyHtml  = data.blocked
    ? `<p class="blocked-msg">⚠ ${escHtml(data.answer)}</p>`
    : `${sectLine}${interpLine}<p class="answer-text">${escHtml(data.answer)}</p>`;

  el.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div class="card-header-left">
          <div class="card-avatar">☾</div>
          <span class="card-label">جواب (Answer)</span>
        </div>
        <div class="card-meta">
          ${sectBadge}
          ${badge}
          ${data.dry_run ? '<span class="badge badge-dry">dry-run</span>' : ''}
        </div>
      </div>
      <div class="card-body">
        ${bodyHtml}
      </div>
      ${guardHtml}
      ${valHtml}
      <div class="card-footer">
        ${footHtml}
      </div>
    </div>`;

  // Store sources for modal
  el.dataset.sources = JSON.stringify(data.sources || []);

  // Wire sources button
  const srcBtn = el.querySelector('.sources-btn');
  if (srcBtn) {
    srcBtn.addEventListener('click', () =>
      openModal(data.sources || [])
    );
  }

  messages.appendChild(el);
}

/* ── Hybrid /query-all-schools: same shell as PageIndex (stacked sects) ─── */
function renderHybridResults(data) {
  const results = Array.isArray(data.results) ? data.results : [];
  const showInterp =
    data.original_question &&
    data.urdu_question &&
    data.original_question.trim() !== data.urdu_question.trim();
  const interpBlock = showInterp
    ? `<div class="pi-results-note">
         <p class="answer-interp"><em>تلاش بطور:</em> ${escHtml(data.urdu_question)}</p>
       </div>`
    : '';
  const elapsed =
    (results[0] && results[0].elapsed_ms) ??
    data.elapsed_ms ??
    '?';

  const sections = results.map((r, i) => _hybridSectSection(r, i)).join('');

  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot';
  wrap.innerHTML = `
    <div class="pi-results pi-results--hybrid">
      <div class="pi-results-header">
        <span class="pi-results-icon">⚡</span>
        <span class="pi-results-title">Hybrid RAG — تین مسالک</span>
        <span class="pi-results-stats">
          ⏱ ${elapsed} ms
          ${data.dry_run ? ' · dry-run' : ''}
        </span>
      </div>
      ${interpBlock}
      ${sections}
    </div>`;

  results.forEach((r, i) => {
    const sec = wrap.querySelector(`.pi-hybrid-section[data-idx="${i}"]`);
    if (!sec) return;
    const btn = sec.querySelector('button.sources-btn');
    if (btn) {
      btn.addEventListener('click', () => openModal(r.sources || []));
    }
  });

  messages.appendChild(wrap);
}

function _hybridSectSection(r, idx) {
  const bySect = {
    deobandi:    { en: 'Banuri',     maslak: 'Deobandi',     maslakUr: 'دیوبندی' },
    barelvi:     { en: 'UrduFatwa',  maslak: 'Barelvi',      maslakUr: 'بریلوی' },
    ahle_hadith: { en: 'A.Hadees',   maslak: 'Ahle Hadees',  maslakUr: 'اہل حدیث' },
  };
  const lbl = bySect[r.sect] || {
    en:         r.maslak || '—',
    maslak:     r.maslak || '',
    maslakUr:   '',
  };
  const label = (r.source_label && String(r.source_label)) || lbl.en;
  const schoolLine = r.maslak
    ? `${r.maslak}${lbl.maslakUr ? ' · ' + lbl.maslakUr : ''}`
    : `${lbl.maslak}${lbl.maslakUr ? ' · ' + lbl.maslakUr : ''}`;

  const initial = (lbl.en && lbl.en[0] ? lbl.en[0] : (label[0] || '?')).toUpperCase();
  const nSrc = (r.sources && r.sources.length) || 0;
  const bad = r.no_match || (r.answer && /^\(retrieval failed\)/i.test(String(r.answer)));
  const hasAns =
    r.answer && String(r.answer).trim() && r.answer !== '(no answer)';

  if (bad && !hasAns) {
    return `
      <div class="pi-section-school pi-hybrid-section pi-section-school--empty" data-sect="${escHtml(r.sect)}" data-idx="${idx}">
        <div class="pi-school-bar">
          <div class="pi-school-avatar">${escHtml(initial)}</div>
          <div class="pi-school-info">
            <span class="pi-school-name">${escHtml(label)}</span>
            <span class="pi-school-maslak">${escHtml(schoolLine)}</span>
          </div>
          <div class="pi-school-status">استرجال ممکن نہیں</div>
        </div>
      </div>`;
  }

  const ans = hasAns ? r.answer : (bad ? r.answer : '—');

  return `
    <div class="pi-section-school pi-hybrid-section" data-sect="${escHtml(r.sect)}" data-idx="${idx}">
      <div class="pi-school-bar">
        <div class="pi-school-avatar">${escHtml(initial)}</div>
        <div class="pi-school-info">
          <span class="pi-school-name">${escHtml(label)}</span>
          <span class="pi-school-maslak">${escHtml(schoolLine)}</span>
        </div>
        <div class="pi-school-count">${nSrc ? escHtml(String(nSrc)) + ' chunks' : '—'}</div>
      </div>
      <div class="pi-fatwa-list">
        <div class="pi-fatwa-item pi-fatwa-item--primary">
          <div class="pi-hybrid-answer">
            <span class="pi-fatwa-alabel">مولف شدہ جواب:</span>
            <p class="pi-hybrid-answer-txt">${escHtml(ans || '')}</p>
          </div>
          <div class="pi-fatwa-footer">
            <div class="footer-stats"></div>
            ${
              nSrc
                ? `<button type="button" class="sources-btn">ماخذ (${nSrc})</button>`
                : ''
            }
          </div>
        </div>
      </div>
    </div>`;
}

function appendErrorBubble(msg) {
  const el = document.createElement('div');
  el.className = 'msg msg-bot';
  el.innerHTML = `
    <div class="card">
      <div class="card-body">
        <p class="blocked-msg">❌ ${escHtml(msg)}</p>
      </div>
    </div>`;
  messages.appendChild(el);
}

/* ── Sub-component builders ─────────────────────────────────────────────── */
function buildStatusBadge(data) {
  if (data.blocked) return '<span class="badge badge-block">BLOCKED</span>';
  if (!data.answer)  return '<span class="badge badge-warn">EMPTY</span>';
  return '<span class="badge badge-ok">OK</span>';
}

function buildGuardHits(hits) {
  if (!hits.length) return '';
  const chips = hits.map(h => `<span class="guard-chip">${escHtml(h)}</span>`).join('');
  return `<div class="guard-hits">${chips}</div>`;
}

function buildValBar(v) {
  if (!v) return '';
  const gDot  = scoreDot(v.grounding, 70, 45);
  const uDot  = scoreDot(v.urdu,      70, 45);
  const hDot  = hazardDot(v.halluc,   10, 25);
  const issues = (v.issues || [])
    .map(i => `<span class="val-issue">${escHtml(i)}</span>`).join(' ');

  return `
    <div class="val-bar">
      <span class="val-score"><span class="val-dot ${gDot}"></span>Ground ${v.grounding}%</span>
      <span class="val-score"><span class="val-dot ${uDot}"></span>Urdu ${v.urdu}%</span>
      <span class="val-score"><span class="val-dot ${hDot}"></span>Halluc ${v.halluc}%</span>
      ${issues}
    </div>`;
}

function scoreDot(val, hi, lo)  {
  return val >= hi ? 'dot-green' : val >= lo ? 'dot-yellow' : 'dot-red';
}
function hazardDot(val, lo, hi) {
  return val <= lo ? 'dot-green' : val <= hi ? 'dot-yellow' : 'dot-red';
}

function buildFooter(data) {
  const srcs = data.sources || [];
  const srcLabel = `ماخذ (${srcs.length})`;
  const srcBtn = srcs.length
    ? `<button class="sources-btn">${srcLabel}</button>` : '';

  const stats = [
    data.elapsed_ms  != null ? `⏱ ${data.elapsed_ms} ms`       : '',
    data.num_chunks  != null ? `📑 ${data.num_chunks} chunks`   : '',
    data.guard_hits?.length  ? `🛡 ${data.guard_hits.length} guards` : '',
  ].filter(Boolean).join(' · ');

  return `
    <div class="footer-stats">${escHtml(stats)}</div>
    ${srcBtn}`;
}

function buildSourceCard(src, rank) {
  const ref = src.reference || src.url || '';
  const refHtml = ref
    ? `<p class="src-file"><a href="${escHtml(ref)}" target="_blank" rel="noopener noreferrer">🔗 Open fatwa source</a></p>`
    : '';

  const fileLine = (src.source_file || src.fatwa_no)
    ? `<p class="src-file">${escHtml(src.source_file || src.fatwa_no)}</p>`
    : '';

  const label = src.source_name || src.maslak || src.category || '—';
  const scoreTxt = (typeof src.score === 'number') ? src.score.toFixed(3) : (src.score ?? '—');
  const SECT_LABELS = { deobandi: 'Deobandi', barelvi: 'Barelvi', ahle_hadith: 'Ahle Hadees' };
  const sectBadge = src.sect
    ? `<span class="src-sect" style="margin-left:6px;padding:2px 8px;border-radius:10px;background:#eef4ff;color:#2247a6;font-size:11px;font-weight:600;">${escHtml(SECT_LABELS[src.sect] || src.sect)}</span>`
    : '';

  const question = (src.question || '').trim();
  const answer   = (src.answer   || '').trim();

  const qBlock = question
    ? `<div class="src-qa-block">
         <span class="src-qa-label">سوال (Question):</span>
         <p class="src-qa-text">${escHtml(question)}</p>
       </div>` : '';

  const aBlock = answer
    ? `<div class="src-qa-block">
         <span class="src-qa-label">جواب (Fatwa):</span>
         <p class="src-qa-text src-qa-answer">${escHtml(answer)}</p>
       </div>` : '';

  return `
    <div class="src-card">
      <div class="src-header">
        <span class="src-rank">#${rank}</span>
        <span class="src-cat">${escHtml(label)}</span>
        ${sectBadge}
        <span class="src-score">score: ${escHtml(scoreTxt)}</span>
      </div>
      ${qBlock}
      ${aBlock}
      ${fileLine}
      ${refHtml}
    </div>`;
}

/* ── History ────────────────────────────────────────────────────────────── */
function addHistory(q) {
  state.history.unshift(q);
  if (state.history.length > 20) state.history.pop();
  renderHistory();
}

function renderHistory() {
  historyList.innerHTML = state.history.map((q, i) =>
    `<li class="history-item" data-idx="${i}" title="${escHtml(q)}">${escHtml(trimWords(q, 6))}</li>`
  ).join('');

  historyList.querySelectorAll('.history-item').forEach(li => {
    li.addEventListener('click', () => {
      const q = state.history[parseInt(li.dataset.idx)];
      if (q) {
        questionInput.value = q;
        questionInput.dispatchEvent(new Event('input'));
        questionInput.focus();
      }
    });
  });
}

/* ── Helpers ────────────────────────────────────────────────────────────── */
function setLoading(on) {
  state.isLoading = on;
  sendBtn.disabled = on;
}

function scrollBottom() {
  requestAnimationFrame(() => {
    messages.scrollTop = messages.scrollHeight;
  });
}

function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function trimWords(str, n) {
  const words = str.split(/\s+/);
  return words.length <= n ? str : words.slice(0, n).join(' ') + '…';
}

/* ────────────────────────────────────────────────────────────────────────
   PageIndex (vectorless) mode
   ──────────────────────────────────────────────────────────────────────── */

const SCHOOL_DISPLAY = {
  Banuri:    { en: 'Banuri',    maslak: 'Deobandi',    maslakUr: 'دیوبندی' },
  fatwaqa:   { en: 'FatwaQA',   maslak: 'Ahle Hadees', maslakUr: 'اہل حدیث' },
  IslamQA:   { en: 'IslamQA',   maslak: 'Ahle Hadees', maslakUr: 'اہل حدیث' },
  urdufatwa: { en: 'UrduFatwa', maslak: 'Barelvi',     maslakUr: 'بریلوی' },
};

async function doPISearch(q) {
  const typingId = appendTyping();
  setLoading(true);

  try {
    const resp = await fetch('/api/search_pageindex', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question: q }),
    });
    removeTyping(typingId);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      appendErrorBubble(err.error || 'PageIndex error');
      return;
    }
    const data = await resp.json();
    renderPIResults(data);
    addHistory(q);
  } catch (err) {
    removeTyping(typingId);
    appendErrorBubble('شبکہ خطا — Network error: ' + err.message);
  } finally {
    setLoading(false);
    scrollBottom();
  }
}

function renderPIResults(data) {
  const results = Array.isArray(data.results) ? data.results : [];

  // Build one full-width section per school (stacked vertically)
  const sections = results.map(r => _piSchoolSection(r)).join('');

  const wrap = document.createElement('div');
  wrap.className = 'msg msg-bot';
  wrap.innerHTML = `
    <div class="pi-results">
      <div class="pi-results-header">
        <span class="pi-results-icon">🌳</span>
        <span class="pi-results-title">PageIndex — چار مسالک</span>
        <span class="pi-results-stats">
          ⏱ ${data.elapsed_ms ?? '?'} ms
          ${data.dry_run ? ' · dry-run' : ''}
        </span>
      </div>
      ${sections}
    </div>`;

  // Wire خلاصہ buttons
  wrap.querySelectorAll('.pi-summarise-btn').forEach(btn => {
    btn.addEventListener('click', () => onSummariseClick(btn));
  });

  messages.appendChild(wrap);
}

/* ── Relevance badge (rank-based: #1=95%, #2=80%, #3=65%, #4=50%) ─── */
function _relTier(pct) {
  const n = Number(pct || 0);
  if (n >= 90) return { cls: 'pi-rel-very-high', label: 'سب سے متعلقہ' };
  if (n >= 75) return { cls: 'pi-rel-high',      label: 'بہت متعلقہ' };
  if (n >= 60) return { cls: 'pi-rel-mid',       label: 'متعلقہ' };
  return                { cls: 'pi-rel-low',       label: 'ممکنہ متعلقہ' };
}

/* ── Full school section (one per madhab, stacked vertically) ──────── */
function _piSchoolSection(r) {
  const lbl = SCHOOL_DISPLAY[r && r.school_id] || { en: '?', maslak: '', maslakUr: '' };
  const maslak  = r && (r.maslak || lbl.maslak) || '';
  const initial = (lbl.en || '?').charAt(0).toUpperCase();

  if (!r || !r.answer_text) {
    return `
      <div class="pi-section-school pi-section-school--empty" data-school="${escHtml(r ? r.school_id : '')}">
        <div class="pi-school-bar">
          <div class="pi-school-avatar">${escHtml(initial)}</div>
          <div class="pi-school-info">
            <span class="pi-school-name">${escHtml(lbl.en)}</span>
            <span class="pi-school-maslak">${escHtml(maslak)}${lbl.maslakUr ? ' · ' + escHtml(lbl.maslakUr) : ''}</span>
          </div>
          <div class="pi-school-status">فتویٰ نہیں ملا</div>
        </div>
      </div>`;
  }

  const fatawa = Array.isArray(r.fatawa) && r.fatawa.length
    ? r.fatawa
    : [{ fatwa_id: r.fatwa_id, fatwa_no: r.fatwa_no, category: r.category,
         subtopic: r.subtopic, query_text: r.query_text, question_text: r.question_text,
         answer_text: r.answer_text, url: r.url, relevance_pct: r.relevance_pct }];

  const fatwaItems = fatawa.map((f, i) => _piFatwaItem(f, i + 1, fatawa.length)).join('');

  return `
    <div class="pi-section-school" data-school="${escHtml(r.school_id)}">
      <div class="pi-school-bar">
        <div class="pi-school-avatar">${escHtml(initial)}</div>
        <div class="pi-school-info">
          <span class="pi-school-name">${escHtml(lbl.en)}</span>
          <span class="pi-school-maslak">${escHtml(maslak)}${lbl.maslakUr ? ' · ' + escHtml(lbl.maslakUr) : ''}</span>
        </div>
        <div class="pi-school-count">${fatawa.length} fatawa</div>
      </div>
      <div class="pi-fatwa-list">
        ${fatwaItems}
      </div>
    </div>`;
}

function _piFatwaItem(f, rank, total) {
  const pct = Number(f.relevance_pct || 0);
  const tier = _relTier(pct);
  const fid = escHtml(f.fatwa_id || '');
  const isFirst = rank === 1;

  const urlHtml = f.url
    ? `<a class="pi-fatwa-url" href="${escHtml(f.url)}" target="_blank" rel="noopener noreferrer">
         🔗 ${escHtml(f.url.length > 80 ? f.url.substring(0, 80) + '…' : f.url)}
       </a>`
    : '<span class="pi-fatwa-url pi-fatwa-url--none">URL not available</span>';

  return `
    <div class="pi-fatwa-item ${isFirst ? 'pi-fatwa-item--primary' : ''}" data-rank="${rank}">
      <div class="pi-fatwa-header">
        <span class="pi-fatwa-rank">${rank}</span>
        <span class="pi-rel-badge ${tier.cls}">${pct}% ${escHtml(tier.label)}</span>
        <span class="pi-fatwa-crumb">
          ${f.category ? escHtml(f.category) : ''}${f.subtopic ? ' › ' + escHtml(f.subtopic) : ''}
        </span>
      </div>

      <div class="pi-fatwa-title">${escHtml(f.query_text || f.fatwa_no || '—')}</div>

      <div class="pi-fatwa-q">
        <span class="pi-fatwa-qlabel">سوال:</span>
        ${escHtml(f.question_text || '—')}
      </div>

      <div class="pi-fatwa-a ${isFirst ? '' : 'pi-fatwa-a--collapsed'}">
        <span class="pi-fatwa-alabel">جواب:</span>
        ${escHtml(f.answer_text || '')}
      </div>
      ${!isFirst ? '<button class="pi-expand-btn" onclick="this.previousElementSibling.classList.toggle(\'pi-fatwa-a--collapsed\'); this.textContent = this.previousElementSibling.classList.contains(\'pi-fatwa-a--collapsed\') ? \'مکمل جواب دکھائیں ▼\' : \'جواب چھپائیں ▲\'">مکمل جواب دکھائیں ▼</button>' : ''}

      <div class="pi-fatwa-footer">
        ${urlHtml}
        <button class="pi-summarise-btn" data-fatwa-id="${fid}">✨ خلاصہ</button>
      </div>
    </div>`;
}

async function onSummariseClick(btn) {
  const fatwaId = btn.dataset.fatwaId;
  if (!fatwaId) return;
  btn.disabled = true;
  btn.textContent = 'لوڈ ہو رہا…';
  try {
    const resp = await fetch('/api/summarise', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ fatwa_id: fatwaId }),
    });
    if (!resp.ok) {
      btn.textContent = 'خرابی';
      return;
    }
    const data = await resp.json();
    const item = btn.closest('.pi-fatwa-item') || btn.closest('.pi-section-school');
    if (!item) return;
    const block = document.createElement('div');
    block.className = 'pi-summary-block';
    block.textContent = data.summary || '(خلاصہ دستیاب نہیں)';
    const existing = item.querySelector('.pi-summary-block');
    if (existing) existing.remove();
    const answerEl = item.querySelector('.pi-fatwa-a') || item.querySelector('.pi-answer');
    if (answerEl) answerEl.after(block);
    btn.textContent = '✓ خلاصہ';
  } catch (err) {
    btn.textContent = 'خرابی';
    btn.disabled = false;
  }
}

