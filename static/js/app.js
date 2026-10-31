const searchForm = document.getElementById('search-form');
const queryInput = document.getElementById('query-input');
const resultsSection = document.getElementById('results-section');
const resultsContainer = document.getElementById('results-container');
const resultsCount = document.getElementById('results-count');
const emptyState = document.getElementById('empty-state');
const loadingState = document.getElementById('loading-state');
const errorState = document.getElementById('error-state');
const errorMessage = document.getElementById('error-message');

const chatModal = document.getElementById('chat-modal');
const chatPaperTitle = document.getElementById('chat-paper-title');
const chatClose = document.getElementById('chat-close');
const chatMessages = document.getElementById('chat-messages');
const chatLoading = document.getElementById('chat-loading');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');

let activeChat = {
    paperId: null,
    paperUrl: null,
    history: [],
    isOpen: false
};

searchForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const query = queryInput.value.trim();
    if (!query) {
        queryInput.focus();
        return;
    }

    showLoading();

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                top_k: 10
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Search failed');
        }

        renderResults(data.results || []);
    } catch (error) {
        showError(error.message);
    }
});

function showLoading() {
    emptyState.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorState.classList.add('hidden');
    loadingState.classList.remove('hidden');
}

function showError(message) {
    loadingState.classList.add('hidden');
    emptyState.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorMessage.textContent = message;
    errorState.classList.remove('hidden');
}

function renderResults(results) {
    loadingState.classList.add('hidden');
    errorState.classList.add('hidden');

    if (!results.length) {
        resultsContainer.innerHTML = '';
        resultsSection.classList.add('hidden');
        emptyState.classList.remove('hidden');
        document.querySelector('.empty-text').textContent = 'No papers found for this query';
        return;
    }

    emptyState.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    resultsCount.textContent = `${results.length} papers`;
    resultsContainer.innerHTML = results.map(renderPaperCard).join('');
}

// Chat functionality

async function openChat(paperId, title, url) {
    if (!paperId) return;

    activeChat.paperId = paperId;
    activeChat.paperUrl = url;
    activeChat.isOpen = true;

    chatPaperTitle.textContent = title || 'Paper Chat';
    chatPaperTitle.title = title || 'Paper Chat';
    chatMessages.innerHTML = '';
    chatModal.classList.remove('hidden');
    chatLoading.classList.remove('hidden');

    chatInput.disabled = true;
    const submitBtn = chatForm.querySelector('button[type="submit"]');
    if (submitBtn) submitBtn.disabled = true;

    try {
        const response = await fetch('/api/chat/open', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_id: paperId, pdf_url: url })
        });

        const data = await response.json();
        chatLoading.classList.add('hidden');

        if (!response.ok) {
            throw new Error(data.error || 'Failed to initialize chat context');
        }

        chatInput.disabled = false;
        if (submitBtn) submitBtn.disabled = false;
        chatMessages.innerHTML = `
            <div class="assistant-message">
                <b>AI Assistant:</b> This paper has been indexed and is ready. What would you like to know?
            </div>
        `;
        chatInput.focus();
    } catch (error) {
        chatLoading.classList.add('hidden');
        chatMessages.innerHTML = `
            <div class="assistant-message error">
                <b>Could not load paper:</b> ${escapeHtml(error.message)}
            </div>
        `;
    }
}

function closeChat() {
    chatModal.classList.add('hidden');
    if (activeChat.paperId) {
        fetch('/api/chat/close', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_id: activeChat.paperId })
        }).catch(() => {});
    }
    activeChat.isOpen = false;
    activeChat.paperId = null;
    activeChat.paperUrl = null;
}

chatClose.addEventListener('click', closeChat);

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && activeChat.isOpen) {
        closeChat();
    }
});

chatModal.addEventListener('click', (e) => {
    if (e.target === chatModal) {
        closeChat();
    }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message || !activeChat.paperId) return;

    chatMessages.innerHTML += `<div class="user-message"><b>You:</b> ${escapeHtml(message)}</div>`;
    chatInput.value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const thinkingId = 'thinking-' + Date.now();
    chatMessages.innerHTML += `<div id="${thinkingId}" class="assistant-message thinking">Thinking...</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;

    chatInput.disabled = true;
    const submitBtn = chatForm.querySelector('button[type="submit"]');
    if (submitBtn) submitBtn.disabled = true;

    try {
        const response = await fetch('/api/chat/message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_id: activeChat.paperId, message })
        });

        const data = await response.json();
        const thinkingElem = document.getElementById(thinkingId);
        if (thinkingElem) thinkingElem.remove();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to get answer');
        }

        const formatted = formatMarkdown(data.response || 'No response available.');
        chatMessages.innerHTML += `<div class="assistant-message"><b>AI Assistant:</b><br>${formatted}</div>`;
    } catch (error) {
        const thinkingElem = document.getElementById(thinkingId);
        if (thinkingElem) thinkingElem.remove();
        chatMessages.innerHTML += `<div class="assistant-message error"><b>Error:</b> ${escapeHtml(error.message)}</div>`;
    } finally {
        chatInput.disabled = false;
        if (submitBtn) submitBtn.disabled = false;
        chatInput.focus();
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});

// Event delegation for "Chat with paper" buttons
resultsContainer.addEventListener('click', (e) => {
    const btn = e.target.closest('.chat-button');
    if (!btn) return;
    const paperId = btn.dataset.paperId;
    const url = btn.dataset.url;
    const card = btn.closest('.paper-card');
    const title = card ? card.querySelector('.paper-title a').textContent.trim() : 'Paper Chat';
    openChat(paperId, title, url);
});

// Expose openChat globally for inline handlers if any
window.openChat = openChat;

function renderPaperCard(paper) {
    const rank = escapeHtml(paper.rank ?? '');
    const title = escapeHtml(paper.title || 'Untitled paper');
    const authors = escapeHtml(paper.authors || 'Authors unavailable');
    const abstract = escapeHtml(paper.abstract || 'No abstract available.');
    const paperId = escapeHtml(paper.paper_id || '');
    const url = safeUrl(paper.url);

    return `
        <article class="paper-card">
            <div class="paper-rank">
                <span class="rank-number">${rank}</span>
            </div>
            <h3 class="paper-title"><a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a></h3>
            <p class="paper-authors">${authors}</p>
            <p class="paper-abstract">${abstract}</p>
            <div class="paper-meta">
                <button type="button" class="chat-button" data-paper-id="${paperId}" data-url="${url}">Chat with paper</button>
            </div>
        </article>
    `;
}

function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}

function formatScore(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(4) : '—';
}

function formatRank(value) {
    const number = Number(value);
    return Number.isInteger(number) && number > 0 ? number : '—';
}

function formatDate(value) {
    if (!value) return '';

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);

    return new Intl.DateTimeFormat(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    }).format(date);
}

function safeUrl(value) {
    if (!value) return '';

    try {
        const url = new URL(value, window.location.origin);
        if (url.protocol === 'http:' || url.protocol === 'https:') {
            return escapeHtml(url.href);
        }
    } catch {
        return '';
    }

    return '';
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}
