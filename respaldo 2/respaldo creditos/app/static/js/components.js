// Reusable UI Components

// ============================================
// TOAST NOTIFICATIONS
// ============================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed top-4 right-4 px-6 py-4 rounded-lg shadow-lg text-white z-50 fade-in ${type === 'success' ? 'bg-green-600' :
        type === 'error' ? 'bg-red-600' :
            type === 'warning' ? 'bg-yellow-600' :
                'bg-blue-600'
        }`;
    toast.textContent = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================
// LOADING SPINNER
// ============================================

function showLoading(container) {
    const spinner = document.createElement('div');
    spinner.className = 'flex justify-center items-center py-8';
    spinner.innerHTML = '<div class="spinner"></div>';
    container.innerHTML = '';
    container.appendChild(spinner);
}

function hideLoading(container) {
    const spinner = container.querySelector('.spinner');
    if (spinner) {
        spinner.parentElement.remove();
    }
}

// ============================================
// PROFESSIONAL CARD COMPONENT
// ============================================

function createProfessionalCard(professional) {
    return `
        <div class="bg-white rounded-lg shadow-md hover:shadow-xl transition p-6">
            <div class="flex items-start space-x-4">
                <img src="${professional.profile_image || '/static/img/default-avatar.png'}" 
                     alt="${professional.full_name}"
                     class="w-20 h-20 rounded-full object-cover">
                <div class="flex-1">
                    <h3 class="text-xl font-semibold text-gray-800">${professional.full_name}</h3>
                    <p class="text-blue-600 font-medium">${professional.specialty || 'Profesional'}</p>
                    <p class="text-gray-600 text-sm">${professional.city || ''}</p>
                    <div class="flex items-center mt-2">
                        <span class="text-yellow-500">★</span>
                        <span class="ml-1 font-semibold">${professional.rating?.toFixed(1) || '0.0'}</span>
                        <span class="ml-1 text-gray-500 text-sm">(${professional.total_reviews || 0} reseñas)</span>
                    </div>
                </div>
            </div>
            <p class="mt-4 text-gray-600 text-sm line-clamp-2">${professional.bio || ''}</p>
            <div class="mt-4 flex space-x-2">
                <a href="/profesional?id=${professional.id}" 
                   class="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition text-center">
                    Ver Perfil
                </a>
                <button onclick="openChatbot(${professional.id})" 
                        class="px-4 py-2 border border-blue-600 text-blue-600 rounded hover:bg-blue-50 transition">
                    Chat
                </button>
            </div>
        </div>
    `;
}

// ============================================
// COMMENT CARD COMPONENT
// ============================================

function createCommentCard(comment) {
    const stars = '★'.repeat(comment.rating) + '☆'.repeat(5 - comment.rating);

    return `
        <div class="bg-white rounded-lg shadow p-6">
            <div class="flex items-start justify-between">
                <div>
                    <h4 class="font-semibold text-gray-800">${comment.author || 'Usuario'}</h4>
                    <div class="text-yellow-500 text-lg">${stars}</div>
                </div>
                <span class="text-sm text-gray-500">${formatDate(comment.created_at)}</span>
            </div>
            <p class="mt-3 text-gray-700">${comment.content}</p>
        </div>
    `;
}

// ============================================
// MODAL COMPONENT
// ============================================

function createModal(id, title, content, actions = '') {
    return `
        <div id="${id}" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50">
            <div class="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
                <div class="px-6 py-4 border-b flex justify-between items-center">
                    <h3 class="text-xl font-semibold">${title}</h3>
                    <button onclick="closeModal('${id}')" class="text-gray-500 hover:text-gray-700">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>
                <div class="px-6 py-4">
                    ${content}
                </div>
                ${actions ? `<div class="px-6 py-4 border-t flex justify-end space-x-2">${actions}</div>` : ''}
            </div>
        </div>
    `;
}

function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

// ============================================
// PAGINATION COMPONENT
// ============================================

function createPagination(currentPage, totalPages, onPageChange) {
    if (totalPages <= 1) return '';

    let html = '<div class="flex justify-center items-center space-x-2 mt-8">';

    // Previous button
    html += `
        <button onclick="${onPageChange}(${currentPage - 1})" 
                ${currentPage === 1 ? 'disabled' : ''}
                class="px-4 py-2 border rounded ${currentPage === 1 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-100'}">
            Anterior
        </button>
    `;

    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
        if (i === currentPage) {
            html += `<span class="px-4 py-2 bg-blue-600 text-white rounded">${i}</span>`;
        } else if (i === 1 || i === totalPages || (i >= currentPage - 2 && i <= currentPage + 2)) {
            html += `<button onclick="${onPageChange}(${i})" class="px-4 py-2 border rounded hover:bg-gray-100">${i}</button>`;
        } else if (i === currentPage - 3 || i === currentPage + 3) {
            html += '<span class="px-2">...</span>';
        }
    }

    // Next button
    html += `
        <button onclick="${onPageChange}(${currentPage + 1})" 
                ${currentPage === totalPages ? 'disabled' : ''}
                class="px-4 py-2 border rounded ${currentPage === totalPages ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-100'}">
            Siguiente
        </button>
    `;

    html += '</div>';
    return html;
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString('es-MX', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
}

function formatCurrency(amount) {
    return new Intl.NumberFormat('es-MX', {
        style: 'currency',
        currency: 'MXN'
    }).format(amount);
}

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validatePassword(password) {
    if (password.length < 8) {
        return { valid: false, message: 'La contraseña debe tener al menos 8 caracteres' };
    }
    if (!/[A-Z]/.test(password)) {
        return { valid: false, message: 'La contraseña debe contener al menos una mayúscula' };
    }
    if (!/\d/.test(password)) {
        return { valid: false, message: 'La contraseña debe contener al menos un número' };
    }
    return { valid: true };
}

// ============================================
// FORM VALIDATION
// ============================================

function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    const inputs = form.querySelectorAll('[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('border-red-500');
            isValid = false;
        } else {
            input.classList.remove('border-red-500');
        }
    });

    return isValid;
}

// ============================================
// CHATBOT WIDGET
// ============================================

let chatbotOpen = false;
let currentProfessionalId = null;
let currentSessionId = null;

function openChatbot(professionalId) {
    currentProfessionalId = professionalId;
    currentSessionId = generateSessionId();

    const widget = document.getElementById('chatbotWidget');
    if (widget) {
        widget.classList.remove('hidden');
        chatbotOpen = true;
        loadChatbotMessages();
    }
}

function closeChatbot() {
    const widget = document.getElementById('chatbotWidget');
    if (widget) {
        widget.classList.add('hidden');
        chatbotOpen = false;
    }
}

function generateSessionId() {
    return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

async function sendChatbotMessage(readResponse = false) {
    const input = document.getElementById('chatbotInput');
    const message = input.value.trim();

    if (!message || !currentProfessionalId) return;

    // Add user message to UI
    addChatMessage('user', message);
    input.value = '';

    // Send to API
    const result = await sendChatMessage(currentProfessionalId, message, currentSessionId);

    if (result.success) {
        addChatMessage('assistant', result.data.message);

        if (readResponse) {
            speakResponse(result.data.message);
        }

        // Show credit warning if present
        if (result.data.warning) {
            showToast(result.data.warning, 'warning');
        }
    } else {
        addChatMessage('assistant', 'Lo siento, hubo un error al procesar tu mensaje.');
        showToast(result.error, 'error');
    }
}

// Voice Command Logic
let isContinuousMode = false;

function toggleContinuousMode() {
    isContinuousMode = !isContinuousMode;
    const continuousBtn = document.getElementById('continuousBtn');

    if (continuousBtn) {
        if (isContinuousMode) {
            continuousBtn.classList.add('bg-purple-600', 'text-white');
            continuousBtn.classList.remove('bg-gray-200', 'text-gray-700');
            showToast('Modo Conversación Continua: ACTIVADO', 'success');
            startVoiceRecognition(); // Start immediately
        } else {
            continuousBtn.classList.remove('bg-purple-600', 'text-white');
            continuousBtn.classList.add('bg-gray-200', 'text-gray-700');
            showToast('Modo Conversación Continua: DESACTIVADO', 'info');
            window.speechSynthesis.cancel(); // Stop speaking if active
        }
    }
}

function startVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window)) {
        alert('Tu navegador no soporta reconocimiento de voz. Prueba con Chrome.');
        return;
    }

    const recognition = new webkitSpeechRecognition();
    recognition.lang = 'es-ES';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    const micBtn = document.getElementById('micBtn');
    if (micBtn) {
        micBtn.classList.add('bg-red-500', 'text-white');
        micBtn.classList.remove('bg-gray-200', 'text-gray-700');
    }

    recognition.start();

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        const input = document.getElementById('chatbotInput');
        if (input) {
            input.value = transcript;
            sendChatbotMessage(true); // Send with readResponse = true
        }
    };

    recognition.onend = function () {
        if (micBtn) {
            micBtn.classList.remove('bg-red-500', 'text-white');
            micBtn.classList.add('bg-gray-200', 'text-gray-700');
        }
        // NOTE: In continuous mode, we don't restart here. We restart AFTER the AI finishes speaking.
    };

    recognition.onerror = function (event) {
        console.error('Speech recognition error', event.error);
        if (micBtn) {
            micBtn.classList.remove('bg-red-500', 'text-white');
            micBtn.classList.add('bg-gray-200', 'text-gray-700');
        }
        // If error in continuous mode, maybe try again? For now, let's stop to avoid infinite error loops.
        if (isContinuousMode && event.error === 'no-speech') {
            // Optional: restart if just silence, but be careful of loops
        }
    };
}

function speakResponse(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'es-ES';

    utterance.onend = function () {
        if (isContinuousMode) {
            // Small delay to avoid picking up system audio
            setTimeout(() => {
                startVoiceRecognition();
            }, 500);
        }
    };

    window.speechSynthesis.speak(utterance);
}

function addChatMessage(role, content) {
    const messagesContainer = document.getElementById('chatbotMessages');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `mb-4 ${role === 'user' ? 'text-right' : 'text-left'}`;
    messageDiv.innerHTML = `
        <div class="inline-block px-4 py-2 rounded-lg ${role === 'user'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-200 text-gray-800'
        }">
            ${content}
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function loadChatbotMessages() {
    if (!currentProfessionalId) return;

    const result = await getChatHistory(currentProfessionalId, currentSessionId);

    if (result.success && result.data.messages) {
        const messagesContainer = document.getElementById('chatbotMessages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
            result.data.messages.reverse().forEach(msg => {
                addChatMessage(msg.role, msg.content);
            });
        }
    }
}
