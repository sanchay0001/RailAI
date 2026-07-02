/**
 * app.js
 * -------
 * All frontend logic for RailAI. Responsibilities:
 *   1. On page load: fetch departments and knowledge base stats from the API
 *   2. Handle question submission: call POST /chat, render the response
 *   3. Render message bubbles (user + AI) with typing animation
 *   4. Display source references after each AI answer
 *   5. Handle PDF upload via POST /upload
 *   6. Handle re-index via POST /reindex
 *
 * No external JS libraries — plain vanilla JS only.
 * The API base URL is auto-detected from window.location so this works
 * both locally (localhost:8000) and on Render (your-app.onrender.com).
 */

// ---------------------------------------------------------------------------
// CONFIGURATION
// ---------------------------------------------------------------------------

// Derive the API base URL from the current page's origin so we never
// hardcode "localhost:8000" — on Render, this becomes your Render URL.
const API_BASE = window.location.origin;

// ---------------------------------------------------------------------------
// DOM ELEMENT REFERENCES
// ---------------------------------------------------------------------------
// Grab all the elements we'll manipulate. Done once at the top so we
// don't call getElementById() repeatedly inside event handlers.

const messagesContainer = document.getElementById('messages-container');
const questionInput      = document.getElementById('question-input');
const sendBtn            = document.getElementById('send-btn');
const sendIcon           = document.getElementById('send-icon');
const deptSelect         = document.getElementById('department-select');
const deptBadge          = document.getElementById('dept-badge');
const sourcesPanel       = document.getElementById('sources-panel');
const sourcesList        = document.getElementById('sources-list');
const statusDot          = document.getElementById('status-dot');
const statusText         = document.getElementById('status-text');
const uploadStatus       = document.getElementById('upload-status');
const statChunks         = document.getElementById('stat-chunks');
const statDocs           = document.getElementById('stat-docs');
const docsList           = document.getElementById('docs-list');

// ---------------------------------------------------------------------------
// INITIALISATION — runs when the page finishes loading
// ---------------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  loadDepartments();    // Populate the department dropdown
  loadKnowledgeBase();  // Populate stats and document list
  checkHealth();        // Set the status indicator (online/offline)

  // Update the department badge in the chat header whenever the
  // department dropdown selection changes.
  deptSelect.addEventListener('change', () => {
    const label = deptSelect.options[deptSelect.selectedIndex].text;
    deptBadge.textContent = label;
  });
});

// ---------------------------------------------------------------------------
// API CALLS — load sidebar data
// ---------------------------------------------------------------------------

/**
 * Fetches department list from GET /departments and populates the
 * <select> dropdown. Called once on page load.
 */
async function loadDepartments() {
  try {
    const res  = await fetch(`${API_BASE}/departments`);
    const data = await res.json();

    // Clear the placeholder "Loading..." option.
    deptSelect.innerHTML = '';

    // Add one <option> per department returned by the API.
    data.departments.forEach(dept => {
      const opt   = document.createElement('option');
      opt.value   = dept.key;        // key="" for "All Departments"
      opt.textContent = dept.label;
      deptSelect.appendChild(opt);
    });
  } catch (err) {
    // If the API is unreachable, show a fallback option so the dropdown
    // isn't completely broken — the user can still type questions.
    deptSelect.innerHTML = '<option value="">All Departments</option>';
    console.error('Failed to load departments:', err);
  }
}

/**
 * Fetches indexed document stats from GET /documents and renders
 * the counts and document list in the sidebar.
 */
async function loadKnowledgeBase() {
  try {
    const res  = await fetch(`${API_BASE}/documents`);
    const data = await res.json();

    // Update the two stat cards.
    statChunks.textContent = data.total_chunks.toLocaleString();
    statDocs.textContent   = data.total_documents;

    // Render one small card per indexed document.
    docsList.innerHTML = data.documents
      .filter(doc => doc.num_chunks > 0) // Skip docs with 0 chunks (e.g. scanned PDFs)
      .map(doc => `
        <div class="doc-item">
          <div class="doc-name">
            <span class="dept-tag dept-${doc.department}">${doc.department}</span>
            ${doc.filename.replace(/^[^_]+_/, '')} <!-- Strip department prefix for cleaner display -->
          </div>
          <div class="doc-meta">${doc.num_pages} pages · ${doc.num_chunks} chunks</div>
        </div>
      `)
      .join('');
  } catch (err) {
    console.error('Failed to load knowledge base:', err);
  }
}

/**
 * Calls GET /health to check if the server is reachable and updates
 * the status dot (green pulse = online, red = offline).
 */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      statusDot.className  = 'status-dot online';
      statusText.textContent = 'System Online';
    } else {
      throw new Error('Non-OK response');
    }
  } catch {
    statusDot.className  = 'status-dot offline';
    statusText.textContent = 'Server Offline';
  }
}

// ---------------------------------------------------------------------------
// CHAT — sending questions and rendering responses
// ---------------------------------------------------------------------------

/**
 * Called when the user clicks Send or presses Enter.
 * Reads the input, calls POST /chat, and renders the conversation.
 */
async function sendQuestion() {
  const question = questionInput.value.trim();

  // Ignore empty submissions.
  if (!question) return;

  // Grab the currently selected department key (empty string = all).
  const department = deptSelect.value;

  // Hide the welcome message on first question (it's a sibling in the
  // container, not a separate element — remove it to keep things clean).
  const welcome = messagesContainer.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  // Render the user's question as a right-aligned bubble.
  appendMessage('user', question);

  // Clear the input and reset its height.
  questionInput.value = '';
  autoResize(questionInput);

  // Disable the send button and show a spinner to prevent double-sends.
  setLoading(true);

  // Show the bouncing dots "AI is thinking" indicator.
  const typingEl = appendTypingIndicator();

  try {
    // POST /chat with the question and optional department filter.
    const res = await fetch(`${API_BASE}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ question, department }),
    });

    // Remove the typing indicator regardless of success or failure.
    typingEl.remove();

    if (!res.ok) {
      // Non-2xx response — parse the FastAPI error detail and show it.
      const errData = await res.json();
      appendMessage('ai', `⚠️ ${errData.detail || 'An error occurred. Please try again.'}`);
      return;
    }

    const data = await res.json();

    // Render the AI answer bubble.
    // formatAnswer() converts the raw text into HTML with proper line
    // breaks and bold formatting.
    appendMessage('ai', formatAnswer(data.answer));

    // If the response includes source references, show the sources panel.
    if (data.sources && data.sources.length > 0) {
      renderSources(data.sources, data.department, data.chunks_used);
    } else {
      sourcesPanel.style.display = 'none';
    }

  } catch (err) {
    // Network error — API completely unreachable.
    typingEl.remove();
    appendMessage('ai', '⚠️ Could not reach the server. Please check your connection.');
    console.error('Chat error:', err);
  } finally {
    // Always re-enable the send button after the request finishes.
    setLoading(false);
  }
}

/**
 * Creates and appends a message bubble to the chat container.
 *
 * @param {string} role    'user' or 'ai'
 * @param {string} content HTML string (for AI) or plain text (for user)
 */
function appendMessage(role, content) {
  const isUser = role === 'user';

  // Get the current time for the timestamp below the bubble.
  const now  = new Date();
  const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Build the message HTML structure.
  const div       = document.createElement('div');
  div.className   = `message ${role}`;
  div.innerHTML   = `
    <div class="message-avatar">${isUser ? '👤' : '🤖'}</div>
    <div>
      <div class="message-bubble">${isUser ? escapeHtml(content) : content}</div>
      <div class="message-time">${time}</div>
    </div>
  `;

  messagesContainer.appendChild(div);

  // Auto-scroll to the newest message so the user doesn't have to
  // manually scroll down after every response.
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  return div;
}

/**
 * Appends the bouncing-dots typing indicator. Returns the element so
 * the caller can remove it once the real response arrives.
 */
function appendTypingIndicator() {
  const div     = document.createElement('div');
  div.className = 'typing-indicator';
  div.innerHTML = `
    <div class="message-avatar" style="background:var(--bg-card);border:1px solid var(--border)">🤖</div>
    <div class="typing-dots">
      <span></span><span></span><span></span>
    </div>
  `;
  messagesContainer.appendChild(div);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return div;
}

/**
 * Converts the AI's raw answer text into HTML:
 *   - Double newlines become paragraph breaks
 *   - **bold** markdown is converted to <strong>
 *   - Numbered lines (1. 2. 3.) are preserved with line breaks
 */
function formatAnswer(text) {
  return text
    // Convert **text** to <strong>text</strong>
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Convert double newlines to paragraph breaks
    .replace(/\n\n/g, '</p><p>')
    // Convert single newlines to <br>
    .replace(/\n/g, '<br>')
    // Wrap everything in a paragraph
    .replace(/^/, '<p>')
    .replace(/$/, '</p>');
}

/**
 * Renders the source reference cards in the sources panel.
 *
 * @param {Array}  sources    array of {filename, page, department} objects
 * @param {string} department human-readable department that was searched
 * @param {number} chunks     number of chunks used
 */
function renderSources(sources, department, chunks) {
  sourcesList.innerHTML = sources.map(s => `
    <div class="source-card">
      <span class="source-filename">📄 ${s.filename.replace(/^[^_]+_/, '')}</span>
      <span class="source-page">Page ${s.page} · <span class="dept-tag dept-${s.department}">${s.department}</span></span>
    </div>
  `).join('');

  // Show the panel
  sourcesPanel.style.display = 'block';
  // Scroll the panel into view if needed
  sourcesPanel.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

// ---------------------------------------------------------------------------
// INPUT HELPERS
// ---------------------------------------------------------------------------

/**
 * Handles keyboard events in the textarea:
 *   Enter alone   → send the question
 *   Shift+Enter   → insert a newline (default textarea behaviour)
 */
function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    // Prevent the default newline insertion before sending.
    event.preventDefault();
    sendQuestion();
  }
}

/**
 * Auto-resizes the textarea to fit its content, up to a max height
 * defined in CSS (max-height: 120px). Without this, a multi-line
 * question would overflow the fixed-height textarea.
 */
function autoResize(el) {
  // Reset height to auto so scrollHeight gives us the natural content height.
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

/**
 * Fills the input with an example question and immediately sends it.
 * Called by the example buttons in the welcome screen.
 *
 * @param {HTMLElement} btn  the clicked example button
 */
function sendExample(btn) {
  questionInput.value = btn.textContent.trim();
  sendQuestion();
}

/**
 * Toggles the send button between its normal state and a loading state:
 *   loading=true  → disable button, show spinner
 *   loading=false → enable button, show arrow icon
 */
function setLoading(loading) {
  sendBtn.disabled    = loading;
  sendIcon.innerHTML  = loading
    ? '<div class="spinner"></div>'
    : '➤';
}

/**
 * Escapes HTML special characters in user-typed text before rendering
 * it in a bubble. Prevents XSS if a user types something like <script>.
 */
function escapeHtml(text) {
  const div       = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// UPLOAD — PDF upload handler
// ---------------------------------------------------------------------------

/**
 * Called when a file is selected via the hidden file input.
 * Sends the PDF to POST /upload with the currently selected department.
 *
 * @param {HTMLInputElement} input  the file input element
 */
async function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;

  // Show uploading message immediately so the user knows it's working.
  uploadStatus.style.color = 'var(--accent)';
  uploadStatus.textContent = `⏳ Uploading ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);
  // Use the currently selected department as the upload tag.
  formData.append('department', deptSelect.value || 'operations');

  try {
    const res  = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body:   formData,
      // Do NOT set Content-Type header manually — the browser sets it
      // automatically with the correct multipart boundary string when
      // using FormData.
    });
    const data = await res.json();

    if (res.ok) {
      uploadStatus.style.color = 'var(--accent-green)';
      uploadStatus.textContent = `✅ Uploaded & indexed: ${data.new_chunks} new chunks`;
      // Refresh the sidebar stats to reflect the new document.
      loadKnowledgeBase();
    } else {
      uploadStatus.style.color = 'var(--accent-orange)';
      uploadStatus.textContent = `❌ ${data.detail || 'Upload failed'}`;
    }
  } catch (err) {
    uploadStatus.style.color = 'var(--accent-orange)';
    uploadStatus.textContent = '❌ Upload failed — server unreachable';
  }

  // Clear the file input so the same file can be re-uploaded if needed.
  input.value = '';
  // Auto-clear the status message after 5 seconds.
  setTimeout(() => { uploadStatus.textContent = ''; }, 5000);
}

// ---------------------------------------------------------------------------
// RE-INDEX
// ---------------------------------------------------------------------------

/**
 * Calls POST /reindex to trigger a manual re-scan of the data/ folder.
 * Updates the sidebar stats when complete.
 */
async function handleReindex() {
  const btn = document.getElementById('reindex-btn');
  btn.textContent = '⏳ Re-indexing...';
  btn.disabled    = true;

  try {
    const res  = await fetch(`${API_BASE}/reindex`, { method: 'POST' });
    const data = await res.json();

    uploadStatus.style.color = 'var(--accent-green)';
    uploadStatus.textContent = `✅ ${data.message}`;
    // Refresh sidebar stats to show new chunk counts.
    loadKnowledgeBase();
  } catch (err) {
    uploadStatus.style.color = 'var(--accent-orange)';
    uploadStatus.textContent = '❌ Re-index failed';
  } finally {
    btn.textContent = '🔄 Re-index Documents';
    btn.disabled    = false;
    setTimeout(() => { uploadStatus.textContent = ''; }, 5000);
  }
}