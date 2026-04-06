/* ── State ─────────────────────────────────────────────────────────────── */
const state = {
  selectedCat: 'ALL',
  history: [],
  isLoading: false,
  lastSources: [],
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

  // Show typing indicator
  const typingId = appendTyping();
  setLoading(true);

  try {
    const resp = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question:    q,
        category:    state.selectedCat,
        top_k:       parseInt(topKInput.value) || 5,
        guardrails:  guardrailsToggle.checked,
        validate:    validateToggle.checked,
      }),
    });

    removeTyping(typingId);

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      appendErrorBubble(err.error || 'Server error');
    } else {
      const data = await resp.json();
      appendBotBubble(data);
      addHistory(q);
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
  const guardHtml = buildGuardHits(data.guard_hits || []);
  const valHtml   = buildValBar(data.validation);
  const footHtml  = buildFooter(data);
  const bodyHtml  = data.blocked
    ? `<p class="blocked-msg">⚠ ${escHtml(data.answer)}</p>`
    : `<p class="answer-text">${escHtml(data.answer)}</p>`;

  el.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div class="card-header-left">
          <div class="card-avatar">☾</div>
          <span class="card-label">جواب (Answer)</span>
        </div>
        <div class="card-meta">
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
  return `
    <div class="src-card">
      <div class="src-header">
        <span class="src-rank">${rank}</span>
        <span class="src-cat">${escHtml(src.category || '—')}</span>
        <span class="src-score">score: ${src.score ?? '—'}</span>
      </div>
      <p class="src-q">${escHtml(src.question || '—')}</p>
      <p class="src-file">${escHtml(src.source_file || src.fatwa_no || '')}</p>
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
