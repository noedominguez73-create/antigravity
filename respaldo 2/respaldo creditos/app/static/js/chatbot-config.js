let professionalId = null;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const documentsList = document.getElementById('documentsList');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const docCount = document.getElementById('docCount');
const noDocuments = document.getElementById('noDocuments');

const urlInput = document.getElementById('urlInput');
const specialtyInput = document.getElementById('specialtyInput');
const descriptionInput = document.getElementById('descriptionInput');
const btnAddUrl = document.getElementById('btnAddUrl');
const urlsList = document.getElementById('urlsList');
const noUrls = document.getElementById('noUrls');
const urlCount = document.getElementById('urlCount');

const chatMessages = document.getElementById('testChatMessages');
const chatInput = document.getElementById('testChatInput');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
    if (!requireRole('professional')) return;

    setupUploadEvents(); setupUrlEvents(); await loadProfessionalId(); if (professionalId) { loadConfig(); loadDocuments(); loadUrls(); }
    // Check browser support
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        const micBtn = document.getElementById('micBtn');
        const contBtn = document.getElementById('continuousBtn');
        if (micBtn) {
            micBtn.disabled = true;
            micBtn.title = "Tu navegador no soporta reconocimiento de voz";
            micBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
        if (contBtn) {
            contBtn.disabled = true;
            contBtn.title = "Tu navegador no soporta reconocimiento de voz";
            contBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    }

    // Initialize voice selector
    initializeVoiceSelector();
});

async function loadProfessionalId() {
    try {
        const user = getCurrentUser();
        if (!user) return;

        const response = await fetch(`/api/profesionales?user_id=${user.id}`, {
            headers: { 'Authorization': `Bearer ${getAuthToken()}` }
        });
        const data = await response.json();

        if (data.success && data.data.professionals && data.data.professionals.length > 0) {
            professionalId = data.data.professionals[0].id;
            // Update UI
            const idDisplay = document.querySelector('header span.text-gray-500');
            if (idDisplay) idDisplay.textContent = `ID: ${professionalId}`;
        } else {
            showToast('No se encontró perfil profesional asociado', 'error');
        }
    } catch (e) {
        console.error("Error loading professional ID:", e);
        showToast('Error al cargar perfil profesional', 'error');
    }
}

// --- Configuration ---
function loadConfig() {
    if (!professionalId) return;
    fetch(`/api/chatbot/${professionalId}/config`, {
        headers: { 'Authorization': `Bearer ${getAuthToken()}` }
    })
        .then(r => r.json())
        .then(data => {
            if (data.success && data.data) {
                const config = data.data;
                if (document.getElementById('isActive')) document.getElementById('isActive').checked = config.is_active;
                if (document.getElementById('welcomeMessage')) document.getElementById('welcomeMessage').value = config.welcome_message || '';
                if (document.getElementById('systemPrompt')) document.getElementById('systemPrompt').value = config.system_prompt || '';
                if (document.getElementById('knowledgeBase')) document.getElementById('knowledgeBase').value = config.knowledge_base || '';
            }
        });
}

function saveConfig() {
    if (!professionalId) return;
    const data = {
        is_active: document.getElementById('isActive').checked,
        welcome_message: document.getElementById('welcomeMessage').value,
        system_prompt: document.getElementById('systemPrompt').value,
        knowledge_base: document.getElementById('knowledgeBase').value
    };

    fetch(`/api/chatbot/${professionalId}/config`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getAuthToken()}`
        },
        body: JSON.stringify(data)
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) showToast('Configuración guardada correctamente', 'success');
            else showToast('Error al guardar configuración', 'error');
        })
        .catch(() => showToast('Error de conexión al guardar', 'error'));
}

// --- Documents ---
let pendingFiles = [];

function setupUploadEvents() {
    if (!uploadZone) return;

    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFilesSelect(e.target.files));

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        handleFilesSelect(e.dataTransfer.files);
    });
}

function handleFilesSelect(files) {
    if (!files.length) return;

    // Add to pending queue
    Array.from(files).forEach(file => {
        // Check duplicates
        if (!pendingFiles.some(f => f.name === file.name && f.size === file.size)) {
            pendingFiles.push(file);
        }
    });

    updatePendingFilesUI();
    fileInput.value = ''; // Reset input
}

function updatePendingFilesUI() {
    const container = document.getElementById('pendingFilesContainer');
    const list = document.getElementById('pendingFilesList');

    if (pendingFiles.length === 0) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');
    list.innerHTML = '';

    pendingFiles.forEach((file, index) => {
        const html = `
                <div class="flex items-center justify-between p-2 bg-blue-50 rounded border border-blue-100 text-sm">
                    <span class="truncate max-w-[200px] text-blue-800">${file.name}</span>
                    <button onclick="removePendingFile(${index})" class="text-red-500 hover:text-red-700 font-bold px-2">×</button>
                </div>
            `;
        list.insertAdjacentHTML('beforeend', html);
    });
}

window.removePendingFile = function (index) {
    pendingFiles.splice(index, 1);
    updatePendingFilesUI();
};

window.uploadPendingFiles = async function () {
    if (!pendingFiles.length || !professionalId) return;

    const btn = document.getElementById('btnUploadFiles');
    btn.disabled = true;
    btn.innerHTML = '<span class="animate-spin">⌛</span> Subiendo...';

    uploadProgress.classList.remove('hidden');
    updateProgress(0);

    let successCount = 0;
    let failCount = 0;
    const total = pendingFiles.length;

    for (let i = 0; i < total; i++) {
        const file = pendingFiles[i];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`/api/chatbot/${professionalId}/knowledge-base/upload`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${getAuthToken()}` },
                body: formData
            });
            const data = await response.json();

            if (data.success) {
                successCount++;
            } else {
                failCount++;
                console.error(`Error uploading ${file.name}:`, data.error);
            }
        } catch (e) {
            failCount++;
            console.error(`Error uploading ${file.name}:`, e);
        }

        updateProgress(Math.round(((i + 1) / total) * 100));
    }

    // Finish
    setTimeout(() => {
        uploadProgress.classList.add('hidden');
        btn.disabled = false;
        btn.innerHTML = '<span>☁️</span> Subir Documentos';

        if (successCount > 0) {
            showToast(`${successCount} archivos subidos correctamente`, 'success');
            loadDocuments();
            pendingFiles = [];
            updatePendingFilesUI();
        }

        if (failCount > 0) {
            showToast(`${failCount} archivos fallaron al subir`, 'error');
        }
    }, 500);
};

function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
}

function loadDocuments() {
    if (!professionalId) return;
    fetch(`/api/chatbot/${professionalId}/knowledge-base/documents`, {
        headers: { 'Authorization': `Bearer ${getAuthToken()}` }
    })
        .then(r => r.json())
        .then(data => {
            const docs = data.data.documents || [];
            documentsList.innerHTML = '';
            docCount.textContent = docs.length;

            if (docs.length === 0) {
                noDocuments.style.display = 'block';
            } else {
                noDocuments.style.display = 'none';
                docs.forEach(doc => {
                    const html = `
                        <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition">
                            <div class="flex items-center gap-3 overflow-hidden">
                                <span class="text-xl">📄</span>
                                <div class="min-w-0">
                                    <p class="text-sm font-medium text-gray-800 truncate">${doc.filename}</p>
                                    <p class="text-xs text-gray-500">${doc.size_formatted} • ${new Date(doc.uploaded_at).toLocaleDateString()}</p>
                                </div>
                            </div>
                            <button onclick="deleteDocument(${doc.id})" class="text-red-500 hover:text-red-700 p-1 rounded hover:bg-red-50 transition" title="Eliminar">
                                🗑️
                            </button>
                        </div>
                    `;
                    documentsList.insertAdjacentHTML('beforeend', html);
                });
            }
        });
}

function deleteDocument(docId) {
    if (!confirm('¿Estás seguro de eliminar este documento?')) return;

    fetch(`/api/chatbot/${professionalId}/knowledge-base/documents/${docId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${getAuthToken()}` }
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                showToast('Documento eliminado', 'success');
                loadDocuments();
            } else {
                showToast('Error al eliminar documento', 'error');
            }
        });
}

// --- URLs ---
function setupUrlEvents() {
    if (btnAddUrl) btnAddUrl.addEventListener('click', addUrl);
}

function addUrl(e) {
    if (e) e.preventDefault();
    if (!professionalId) return;

    const url = urlInput.value.trim();
    const specialty = specialtyInput.value.trim();
    const description = descriptionInput.value.trim();

    if (!url) {
        showToast('Por favor ingresa una URL válida', 'warning');
        return;
    }

    btnAddUrl.disabled = true;
    btnAddUrl.innerHTML = '<span class="animate-spin">⌛</span> Agregando...';

    const payload = { url, specialty, description };

    fetch(`/api/profesionales/${professionalId}/urls`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getAuthToken()}`
        },
        body: JSON.stringify(payload)
    })
        .then(r => r.json())
        .then(data => {
            btnAddUrl.disabled = false;
            btnAddUrl.innerHTML = '<span>➕</span> Agregar URL';

            if (data.success) {
                urlInput.value = '';
                specialtyInput.value = '';
                descriptionInput.value = '';
                loadUrls();
                showToast('URL agregada exitosamente', 'success');
            } else {
                showToast(data.error || 'Error al agregar URL', 'error');
            }
        })
        .catch(err => {
            btnAddUrl.disabled = false;
            btnAddUrl.innerHTML = '<span>➕</span> Agregar URL';
            showToast('Error de conexión', 'error');
        });
}

function loadUrls() {
    if (!professionalId) return;

    fetch(`/api/profesionales/${professionalId}/urls`, {
        headers: { 'Authorization': `Bearer ${getAuthToken()}` }
    })
        .then(r => r.json())
        .then(data => {
            const urls = data.data.urls || [];
            urlsList.innerHTML = '';

            if (urls.length === 0) {
                noUrls.style.display = 'block';
                urlCount.textContent = '0';
            } else {
                noUrls.style.display = 'none';
                urlCount.textContent = urls.length;
                urls.forEach(url => addUrlUI(url));
            }
        })
        .catch(e => console.error("Error loading URLs:", e));
}

function addUrlUI(url) {
    const date = new Date(url.created_at).toLocaleDateString('es-ES');
    const fetched = url.last_fetched ? new Date(url.last_fetched).toLocaleString('es-ES') : 'Pendiente';

    const html = `
        <div class="p-3 bg-gray-50 rounded-lg border border-gray-200 hover:bg-gray-100 transition">
            <div class="flex justify-between items-start">
                <div class="flex-1 min-w-0 mr-2">
                    <div class="flex items-center gap-2 mb-1">
                        <span class="text-lg">🌐</span>
                        <a href="${url.url}" target="_blank" class="text-blue-600 hover:underline text-sm font-medium truncate block">${url.url}</a>
                    </div>
                    <div class="flex flex-wrap gap-2 text-xs text-gray-500 mb-1">
                        <span class="bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">${url.specialty || 'General'}</span>
                        <span>Agregado: ${date}</span>
                    </div>
                    <div class="text-xs text-gray-400">Consultado: ${fetched}</div>
                    ${url.description ? (() => {
            const div = document.createElement('div');
            div.textContent = url.description;
            return `<div class="text-xs text-gray-600 mt-1 italic border-l-2 border-gray-300 pl-2">${div.innerHTML}</div>`;
        })() : ''}
                </div>
                <div class="flex flex-col gap-1">
                    <button onclick="refreshUrl(${url.id})" class="text-blue-500 hover:text-blue-700 p-1 rounded hover:bg-blue-50" title="Actualizar contenido">🔄</button>
                    <button onclick="deleteUrl(${url.id})" class="text-red-500 hover:text-red-700 p-1 rounded hover:bg-red-50" title="Eliminar">🗑️</button>
                </div>
            </div>
        </div>
        `;
    urlsList.insertAdjacentHTML('beforeend', html);
}

window.deleteUrl = function (urlId) {
    if (!professionalId) return;
    if (!confirm('¿Eliminar esta URL?')) return;

    fetch(`/api/profesionales/${professionalId}/urls/${urlId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${getAuthToken()}` }
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                loadUrls();
                showToast('URL eliminada', 'success');
            }
            else showToast('Error al eliminar URL', 'error');
        });
};

window.refreshUrl = function (urlId) {
    if (!professionalId) return;

    fetch(`/api/profesionales/${professionalId}/urls/${urlId}/refresh`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${getAuthToken()}` }
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                showToast('Contenido actualizado correctamente', 'success');
                loadUrls();
            } else {
                showToast(data.error || 'Error al actualizar', 'error');
            }
        });
};

// --- Test Chat ---
async function sendTestMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage(text, 'user');
    chatInput.value = '';

    const typingId = appendMessage('Escribiendo...', 'bot', true);

    try {
        const payload = {
            message: text,
            professional_id: professionalId
        };

        if (window.currentSessionId) {
            payload.session_id = window.currentSessionId;
        }

        const response = await fetch(`/api/chatbot/${professionalId}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${getAuthToken()}`
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        document.getElementById(typingId).remove();

        if (data.success) {
            if (data.data && data.data.session_id) {
                window.currentSessionId = data.data.session_id;
            }
            const content = data.data ? (data.data.respuesta || data.data.message) : data.response;
            appendMessage(content, 'bot');

            // Speak the response
            if (isContinuousConversationActive) {
                speakText(content);
            }
        } else {
            appendMessage('Error: ' + (data.error || 'No se pudo obtener respuesta'), 'error');
        }
    } catch (e) {
        if (document.getElementById(typingId)) document.getElementById(typingId).remove();
        appendMessage('Error de conexión', 'error');
        console.error(e);
    }
}

function appendMessage(text, type, isTyping = false) {
    const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    const div = document.createElement('div');
    div.id = id;
    div.className = `flex ${type === 'user' ? 'justify-end' : 'justify-start'}`;

    const bubble = document.createElement('div');
    bubble.className = `max-w-[85%] rounded-2xl px-4 py-2 shadow-sm ${type === 'user'
        ? 'bg-blue-600 text-white rounded-br-none'
        : type === 'error'
            ? 'bg-red-100 text-red-600'
            : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none'
        }`;

    if (isTyping) {
        bubble.innerHTML = '<span class="animate-pulse">...</span>';
        bubble.classList.add('italic', 'text-gray-500');
    } else {
        bubble.textContent = text;
    }

    div.appendChild(bubble);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return id;
}

// Auth Helper
function getAuthToken() {
    return localStorage.getItem('token') || '';
}

// Toast Helper
function showToast(message, type = 'info') {
    if (window.showToastComponent) {
        window.showToastComponent(message, type);
        return;
    }
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg shadow-lg text-white z-50 transition-opacity duration-300 ${type === 'success' ? 'bg-green-600' :
        type === 'error' ? 'bg-red-600' : 'bg-blue-600'
        }`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// --- Voice Recognition ---
let recognition = null;
let isContinuousConversationActive = false;
let live_interaction_enabled = false; // Maps to recognition state conceptually
let isBotSpeaking = false; // Flag to prevent self-listening

function initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast('Tu navegador no soporta reconocimiento de voz', 'error');
        return null;
    }
    const r = new SpeechRecognition();
    r.lang = 'es-ES';
    r.continuous = isContinuousConversationActive;
    r.interimResults = false;

    r.onstart = function () {
        live_interaction_enabled = true;
        const btn = document.getElementById('micBtn');
        if (btn) {
            btn.classList.remove('bg-gray-200', 'text-gray-700');
            btn.classList.add('bg-red-100', 'text-red-600', 'animate-pulse');
        }
    };

    r.onend = function () {
        live_interaction_enabled = false;
        const btn = document.getElementById('micBtn');
        if (btn) {
            btn.classList.remove('bg-red-100', 'text-red-600', 'animate-pulse');
            btn.classList.add('bg-gray-200', 'text-gray-700');
        }
        // Only restart if continuous mode is active AND the bot is NOT speaking
        if (isContinuousConversationActive && !isBotSpeaking) {
            try { r.start(); } catch (e) { }
        }
    };

    r.onresult = function (event) {
        // If bot is speaking, ignore results (double safety)
        if (isBotSpeaking) return;

        const transcript = event.results[event.results.length - 1][0].transcript;
        const input = document.getElementById('testChatInput');
        if (input) {
            input.value = transcript;
            if (isContinuousConversationActive) {
                sendTestMessage();
            }
        }
    };

    r.onerror = function (event) {
        console.error("Speech error:", event.error);
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
            showToast('Error voz: ' + event.error, 'error');
        }
    };

    return r;
}

window.startTestVoiceRecognition = function () {
    if (isBotSpeaking) return; // Don't start if speaking

    if (!recognition) recognition = initRecognition();
    if (!recognition) return;

    try {
        recognition.start();
    } catch (e) {
        if (e.name !== 'InvalidStateError') {
            console.error("Recognition start error:", e);
        }
    }
};

window.toggleContinuousConversation = function () {
    isContinuousConversationActive = !isContinuousConversationActive;
    const btn = document.getElementById('continuousBtn');

    if (recognition) {
        recognition.continuous = isContinuousConversationActive;
        recognition.stop();
    }

    if (isContinuousConversationActive) {
        btn.classList.remove('bg-gray-200', 'text-gray-700', 'hover:bg-gray-300');
        btn.classList.add('bg-blue-600', 'text-white', 'hover:bg-blue-700', 'ring-2', 'ring-blue-300');
        showToast('Modo continuo activado', 'success');

        // Only start if not speaking
        if (!isBotSpeaking) {
            setTimeout(() => startTestVoiceRecognition(), 200);
        }
    } else {
        btn.classList.remove('bg-blue-600', 'text-white', 'hover:bg-blue-700', 'ring-2', 'ring-blue-300');
        btn.classList.add('bg-gray-200', 'text-gray-700', 'hover:bg-gray-300');
        showToast('Modo continuo desactivado', 'info');
        // Stop speech if any
        if (window.speechSynthesis) window.speechSynthesis.cancel();
        isBotSpeaking = false;
    }
};

function enableContextTracking() {
    return !!window.currentSessionId;
}

// --- Voice Selection ---
let availableVoices = [];
let selectedVoice = null;

function initializeVoiceSelector() {
    // Load voices (async in some browsers)
    if (window.speechSynthesis) {
        // Try to load immediately
        loadVoices();

        // Also listen for voiceschanged event (Chrome)
        window.speechSynthesis.onvoiceschanged = loadVoices;
    }
}

function loadVoices() {
    const voices = window.speechSynthesis.getVoices();
    availableVoices = rankSpanishVoices(voices);
    populateVoiceSelector();
    loadSavedVoicePreference();
}

function rankSpanishVoices(voices) {
    // Filter Spanish voices
    const spanishVoices = voices.filter(v => v.lang.toLowerCase().includes('es'));

    // Score and sort
    const scored = spanishVoices.map(voice => {
        let score = 0;
        const lang = voice.lang.toLowerCase();

        // Dialect priority
        if (lang.includes('mx')) score += 300; // es-MX highest priority
        else if (lang.includes('us')) score += 200; // es-US second
        else if (lang.includes('es-es')) score += 100; // es-ES third
        else score += 50; // Other Spanish variants

        // Quality bonus for known providers
        const name = voice.name.toLowerCase();
        if (name.includes('google')) score += 20;
        else if (name.includes('microsoft')) score += 15;

        return { voice, score };
    });

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    // Return top 5
    return scored.slice(0, 5).map(s => s.voice);
}

function populateVoiceSelector() {
    const selector = document.getElementById('voiceSelector');
    if (!selector) return;

    // Clear existing options
    selector.innerHTML = '';

    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Automática (mejor disponible)';
    selector.appendChild(defaultOption);

    // Add available voices
    availableVoices.forEach((voice, index) => {
        const option = document.createElement('option');
        option.value = index.toString();

        // Format display name
        const langLabel = voice.lang.includes('mx') ? '🇲🇽 MX' :
            voice.lang.includes('us') ? '🇺🇸 US' :
                voice.lang.includes('es-es') ? '🇪🇸 ES' : '🌎';
        option.textContent = `${langLabel} - ${voice.name}`;

        selector.appendChild(option);
    });
}

function loadSavedVoicePreference() {
    try {
        const saved = localStorage.getItem('selectedVoice');
        if (!saved) return;

        const { name, lang } = JSON.parse(saved);

        // Find matching voice in available voices
        const voiceIndex = availableVoices.findIndex(v =>
            v.name === name && v.lang === lang
        );

        if (voiceIndex !== -1) {
            selectedVoice = availableVoices[voiceIndex];
            const selector = document.getElementById('voiceSelector');
            if (selector) selector.value = voiceIndex.toString();
        }
    } catch (e) {
        console.error('Error loading voice preference:', e);
    }
}

window.handleVoiceChange = function () {
    const selector = document.getElementById('voiceSelector');
    if (!selector) return;

    const index = selector.value;

    if (index === '') {
        // Auto mode
        selectedVoice = null;
        localStorage.removeItem('selectedVoice');
    } else {
        const voiceIndex = parseInt(index);
        selectedVoice = availableVoices[voiceIndex];

        // Save preference
        localStorage.setItem('selectedVoice', JSON.stringify({
            name: selectedVoice.name,
            lang: selectedVoice.lang
        }));
    }
};

// --- Text to Speech ---
function speakText(text) {
    if (!isContinuousConversationActive) return;
    if (!window.speechSynthesis) return;

    // Set flag BEFORE stopping recognition
    isBotSpeaking = true;

    // Cancel previous speech
    window.speechSynthesis.cancel();

    // Stop recognition to avoid hearing itself
    if (recognition) {
        recognition.stop();
        // Note: this triggers onend, but !isBotSpeaking check will prevent restart
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-ES';

    // Voice selection with priority fallback
    if (selectedVoice) {
        // Use user-selected voice
        utterance.voice = selectedVoice;
    } else {
        // Automatic selection: prioritize es-MX -> es-US -> es-ES
        const voices = window.speechSynthesis.getVoices();
        const autoVoice = voices.find(v => v.lang.toLowerCase().includes('es-mx')) ||
            voices.find(v => v.lang.toLowerCase().includes('es-us')) ||
            voices.find(v => v.lang.toLowerCase().includes('es-es')) ||
            voices.find(v => v.lang.toLowerCase().includes('es'));
        if (autoVoice) utterance.voice = autoVoice;
    }

    utterance.onend = function () {
        isBotSpeaking = false;
        if (isContinuousConversationActive && recognition) {
            try { recognition.start(); } catch (e) { }
        }
    };

    // Safety timeout in case onend doesn't fire (e.g. browser bug)
    // Estimate duration: 100 words ~ 30-40 seconds. 
    // Let's just rely on onend for now, but ensure we reset if error.
    utterance.onerror = function () {
        isBotSpeaking = false;
    };

    window.speechSynthesis.speak(utterance);
}
