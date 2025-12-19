// Enhanced Chatbot Widget with Gemini Integration
// Floating widget in bottom-right corner with full functionality

class ChatbotWidget {
    constructor() {
        this.isOpen = false;
        this.professionalId = null;
        this.sessionId = null;
        this.messages = [];
        this.credits = 0;

        this.init();
    }

    init() {
        // Create widget HTML
        this.createWidget();

        // Attach event listeners
        this.attachEventListeners();

        // Check if on professional page
        this.checkProfessionalPage();
    }

    createWidget() {
        const widgetHTML = `
            <!-- Floating Chat Button -->
            <button id="chatbotToggle" class="fixed bottom-6 right-6 w-16 h-16 bg-blue-600 text-white rounded-full shadow-2xl hover:bg-blue-700 transition-all z-40 flex items-center justify-center">
                <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
                </svg>
                <span id="chatbotBadge" class="hidden absolute -top-1 -right-1 w-6 h-6 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">1</span>
            </button>

            <!-- Chat Widget -->
            <div id="chatbotWidget" class="hidden fixed bottom-24 right-6 w-96 bg-white rounded-2xl shadow-2xl z-50 flex flex-col" style="height: 600px; max-height: 80vh;">
                <!-- Header -->
                <div class="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 rounded-t-2xl flex justify-between items-center">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                            <svg class="w-6 h-6 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"></path>
                                <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z"></path>
                            </svg>
                        </div>
                        <div>
                            <h4 class="font-semibold" id="chatbotTitle">Asistente Virtual</h4>
                            <p class="text-xs text-blue-100" id="chatbotSubtitle">Powered by Gemini AI</p>
                        </div>
                    </div>
                    <button id="chatbotClose" class="text-white hover:text-gray-200 transition">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>

                <!-- Credits Display -->
                <div id="creditsBar" class="hidden bg-yellow-50 border-b border-yellow-200 px-6 py-2 flex items-center justify-between">
                    <div class="flex items-center space-x-2">
                        <svg class="w-5 h-5 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M8.433 7.418c.155-.103.346-.196.567-.267v1.698a2.305 2.305 0 01-.567-.267C8.07 8.34 8 8.114 8 8c0-.114.07-.34.433-.582zM11 12.849v-1.698c.22.071.412.164.567.267.364.243.433.468.433.582 0 .114-.07.34-.433.582a2.305 2.305 0 01-.567.267z"></path>
                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-13a1 1 0 10-2 0v.092a4.535 4.535 0 00-1.676.662C6.602 6.234 6 7.009 6 8c0 .99.602 1.765 1.324 2.246.48.32 1.054.545 1.676.662v1.941c-.391-.127-.68-.317-.843-.504a1 1 0 10-1.51 1.31c.562.649 1.413 1.076 2.353 1.253V15a1 1 0 102 0v-.092a4.535 4.535 0 001.676-.662C13.398 13.766 14 12.991 14 12c0-.99-.602-1.765-1.324-2.246A4.535 4.535 0 0011 9.092V7.151c.391.127.68.317.843.504a1 1 0 101.511-1.31c-.563-.649-1.413-1.076-2.354-1.253V5z" clip-rule="evenodd"></path>
                        </svg>
                        <span class="text-sm font-semibold text-yellow-800">Créditos: <span id="creditsCount">0</span></span>
                    </div>
                    <a href="/creditos" class="text-xs text-blue-600 hover:underline">Recargar</a>
                </div>

                <!-- Messages Container -->
                <div id="chatbotMessages" class="flex-1 overflow-y-auto p-6 bg-gray-50 space-y-4">
                    <div class="text-center text-gray-500 text-sm">
                        <p>¡Hola! Soy tu asistente virtual.</p>
                        <p>¿En qué puedo ayudarte hoy?</p>
                    </div>
                </div>

                <!-- Typing Indicator -->
                <div id="typingIndicator" class="hidden px-6 py-2 bg-gray-50">
                    <div class="flex items-center space-x-2 text-gray-500">
                        <div class="flex space-x-1">
                            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 0ms"></div>
                            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 150ms"></div>
                            <div class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style="animation-delay: 300ms"></div>
                        </div>
                        <span class="text-sm">Escribiendo...</span>
                    </div>
                </div>

                <!-- Input Area -->
                <div class="border-t bg-white px-6 py-4 rounded-b-2xl">
                    <div class="flex space-x-2">
                        <input 
                            type="text" 
                            id="chatbotInput" 
                            placeholder="Escribe tu mensaje..." 
                            class="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-600 text-sm"
                            autocomplete="off"
                        >
                        <button 
                            id="chatbotSend" 
                            class="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                            </svg>
                        </button>
                    </div>
                    <p class="text-xs text-gray-500 mt-2">Cada mensaje consume 1 crédito</p>
                </div>
            </div>
        `;

        // Inject into body
        const container = document.createElement('div');
        container.innerHTML = widgetHTML;
        document.body.appendChild(container);
    }

    attachEventListeners() {
        // Toggle widget
        document.getElementById('chatbotToggle').addEventListener('click', () => {
            this.toggle();
        });

        // Close widget
        document.getElementById('chatbotClose').addEventListener('click', () => {
            this.close();
        });

        // Send message on button click
        document.getElementById('chatbotSend').addEventListener('click', () => {
            this.sendMessage();
        });

        // Send message on Enter key
        document.getElementById('chatbotInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    checkProfessionalPage() {
        // Check if we're on a professional page
        const params = new URLSearchParams(window.location.search);
        const profId = params.get('id');

        if (profId && window.location.pathname.includes('profesional')) {
            this.professionalId = parseInt(profId);
            this.loadProfessionalInfo();
        }
    }

    async loadProfessionalInfo() {
        if (!this.professionalId) return;

        try {
            const result = await getProfessional(this.professionalId);
            if (result.success) {
                document.getElementById('chatbotTitle').textContent = result.data.full_name;
                document.getElementById('chatbotSubtitle').textContent = result.data.specialty || 'Profesional';
            }

            // Load chatbot config
            const configResult = await getChatbotConfig(this.professionalId);
            if (configResult.success && configResult.data.is_active) {
                // Show welcome message
                if (configResult.data.welcome_message) {
                    this.addMessage('assistant', configResult.data.welcome_message);
                }
            }

            // Load credits if user is the professional
            const user = getCurrentUser();
            if (user && user.role === 'professional') {
                await this.loadCredits();
            }
        } catch (error) {
            console.error('Error loading professional info:', error);
        }
    }

    async loadCredits() {
        if (!this.professionalId) return;

        try {
            const result = await getCredits(this.professionalId);
            if (result.success) {
                this.credits = result.data.available || 0;
                document.getElementById('creditsCount').textContent = this.credits;
                document.getElementById('creditsBar').classList.remove('hidden');

                // Show warning if low
                if (result.data.warning) {
                    this.addMessage('system', result.data.warning);
                }
            }
        } catch (error) {
            console.error('Error loading credits:', error);
        }
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    open() {
        document.getElementById('chatbotWidget').classList.remove('hidden');
        document.getElementById('chatbotToggle').classList.add('scale-0');
        this.isOpen = true;

        // Generate session ID if not exists
        if (!this.sessionId) {
            this.sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        }

        // Focus input
        setTimeout(() => {
            document.getElementById('chatbotInput').focus();
        }, 100);
    }

    close() {
        document.getElementById('chatbotWidget').classList.add('hidden');
        document.getElementById('chatbotToggle').classList.remove('scale-0');
        this.isOpen = false;
    }

    async sendMessage() {
        const input = document.getElementById('chatbotInput');
        const message = input.value.trim();

        if (!message) return;

        if (!this.professionalId) {
            this.addMessage('system', 'Por favor selecciona un profesional para chatear.');
            return;
        }

        // Check credits
        if (this.credits < 1) {
            this.addMessage('system', 'No tienes créditos suficientes. Por favor recarga tu cuenta.');
            return;
        }

        // Add user message
        this.addMessage('user', message);
        input.value = '';

        // Show typing indicator
        document.getElementById('typingIndicator').classList.remove('hidden');

        // Disable send button
        const sendBtn = document.getElementById('chatbotSend');
        sendBtn.disabled = true;

        try {
            // Send to API
            const result = await sendChatMessage(this.professionalId, message, this.sessionId);

            // Hide typing indicator
            document.getElementById('typingIndicator').classList.add('hidden');
            sendBtn.disabled = false;

            if (result.success) {
                // Add assistant response
                this.addMessage('assistant', result.data.message);

                // Update credits
                if (result.data.credits_remaining !== undefined) {
                    this.credits = result.data.credits_remaining;
                    document.getElementById('creditsCount').textContent = this.credits;
                }

                // Show warning if present
                if (result.data.warning) {
                    this.addMessage('system', result.data.warning);
                }
            } else {
                this.addMessage('system', 'Error: ' + result.error);
            }
        } catch (error) {
            document.getElementById('typingIndicator').classList.add('hidden');
            sendBtn.disabled = false;
            this.addMessage('system', 'Error al enviar mensaje. Por favor intenta de nuevo.');
            console.error('Chat error:', error);
        }
    }

    addMessage(role, content) {
        const container = document.getElementById('chatbotMessages');

        // Clear placeholder if first message
        if (container.querySelector('.text-center')) {
            container.innerHTML = '';
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

        const bubbleClass = role === 'user'
            ? 'bg-blue-600 text-white'
            : role === 'system'
                ? 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                : 'bg-white text-gray-800 border border-gray-200';

        messageDiv.innerHTML = `
            <div class="max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${bubbleClass} shadow-sm">
                <p class="text-sm whitespace-pre-wrap">${this.escapeHtml(content)}</p>
                <p class="text-xs mt-1 opacity-70">${new Date().toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit' })}</p>
            </div>
        `;

        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;

        // Store message
        this.messages.push({ role, content, timestamp: new Date() });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chatbot widget when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatbotWidget = new ChatbotWidget();
});

// Global function to open chatbot (called from professional cards)
function openChatbot(professionalId) {
    if (window.chatbotWidget) {
        window.chatbotWidget.professionalId = professionalId;
        window.chatbotWidget.open();
        window.chatbotWidget.loadProfessionalInfo();
    }
}
