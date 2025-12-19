/**
 * FinanceVoiceForms.js
 * Handles complex voice form filling for Finance App (Tasks, etc.)
 */

const FinanceVoiceForms = {
    recognition: null,
    isListening: false,
    activeField: 'title', // title, desc, priority, date, time
    lastInterimLength: 0,
    callbacks: {
        onFieldChange: null, // (fieldId) => {}
        onSubmit: null,      // () => {}
        onCancel: null       // () => {}
    },

    init(callbacks = {}) {
        this.callbacks = { ...this.callbacks, ...callbacks };

        if ('webkitSpeechRecognition' in window) {
            // eslint-disable-next-line no-undef
            this.recognition = new webkitSpeechRecognition();
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.lang = 'es-ES';

            this.recognition.onstart = () => {
                this.isListening = true;
                this.updateUI(true);
                this.setActiveField('title'); // Start at title
            };

            this.recognition.onend = () => {
                this.isListening = false;
                this.updateUI(false);
            };

            this.recognition.onresult = (event) => this.handleResult(event);
        }
    },

    toggle() {
        if (!this.recognition) {
            alert("Tu navegador no soporta reconocimiento de voz.");
            return;
        }
        if (this.isListening) {
            this.recognition.stop();
        } else {
            this.recognition.start();
        }
    },

    updateUI(listening) {
        const btn = document.getElementById('voice-assistant-btn');
        if (btn) {
            if (listening) btn.classList.add('animate-pulse', 'ring-4', 'ring-amber-300');
            else btn.classList.remove('animate-pulse', 'ring-4', 'ring-amber-300');
        }

        // Auto-open modal if needed
        const modal = document.getElementById('new-task-modal');
        if (listening && modal && modal.classList.contains('hidden')) {
            // Assume global function exists or callback
            if (window.openNewTaskModal) window.openNewTaskModal();
        }
    },

    setActiveField(fieldId) {
        if (this.activeField !== fieldId) {
            this.activeField = fieldId;
            const map = {
                'title': 'task-title',
                'desc': 'task-desc',
                'priority': 'task-priority',
                'date': 'task-date',
                'time': 'task-time'
            };
            const el = document.getElementById(map[fieldId]);
            if (el) {
                el.focus();
                this.lastInterimLength = 0; // Reset tracking new field
                if (this.callbacks.onFieldChange) this.callbacks.onFieldChange(fieldId);
            }
        }
    },

    handleResult(event) {
        let interim = '';
        let finalChunk = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalChunk += event.results[i][0].transcript;
            } else {
                interim += event.results[i][0].transcript;
            }
        }

        finalChunk = finalChunk.toLowerCase();
        interim = interim.toLowerCase();
        let fullText = (finalChunk + " " + interim).trim();

        // Pipeline: 1. Navigation -> 2. Input -> 3. Actions
        this.checkNavigation(fullText);
        this.handleFieldInput(fullText, finalChunk);
        this.checkActions(fullText);
    },

    checkNavigation(str) {
        str = str.trim();
        if (str.includes('prioridad')) this.setActiveField('priority');
        if (str.includes('fecha')) this.setActiveField('date');
        if (str.includes('hora')) this.setActiveField('time');
        if (str.includes('título')) this.setActiveField('title');
        if (str.includes('descripción')) this.setActiveField('desc');

        // Smart "Next"
        if (str.includes('siguiente') || str.includes('continuar')) {
            const order = ['title', 'desc', 'priority', 'date', 'time'];
            const idx = order.indexOf(this.activeField);
            if (idx < order.length - 1) this.setActiveField(order[idx + 1]);
        }
    },

    checkActions(str) {
        str = str.trim();
        if (str.includes('tarea recurrente') || str.includes('recurrente')) {
            const chk = document.getElementById('task-recurring');
            if (chk) chk.checked = true;
        }
        if (str.includes('crear tarea') || str.includes('guardar') || str.includes('listo')) {
            // Callback for submit
            if (this.callbacks.onSubmit) this.callbacks.onSubmit();
            this.toggle(); // Stop listening
        }
        if (str.includes('cancelar') || str.includes('cerrar')) {
            if (this.callbacks.onCancel) this.callbacks.onCancel();
            this.toggle(); // Stop listening
        }
    },

    handleFieldInput(fullText, finalChunk) {
        // Regex to strip ALL commands (Navigation + Actions)
        const commandRegex = /prioridad|fecha|hora|título|descripción|siguiente|continuar|crear tarea|guardar|listo|cancelar|cerrar|tarea recurrente|recurrente/gi;
        const cleanText = fullText.replace(commandRegex, '').trim();

        if (!cleanText && this.lastInterimLength === 0) return;

        // Priority Logic
        if (this.activeField === 'priority') {
            // Assume global setPriority
            if (cleanText.includes('alta') || cleanText.includes('urgente')) window.setPriority('high');
            if (cleanText.includes('media') || cleanText.includes('normal')) window.setPriority('medium');
            return;
        }

        // Date Logic
        if (this.activeField === 'date') {
            const input = document.getElementById('task-date');
            if (!input) return;
            const today = new Date();
            if (cleanText.includes('hoy')) input.valueAsDate = today;
            if (cleanText.includes('mañana')) {
                const tmrw = new Date(today); tmrw.setDate(tmrw.getDate() + 1);
                input.valueAsDate = tmrw;
            }
            if (cleanText.includes('esta semana')) {
                const diff = 5 - today.getDay(); // To Friday
                const nextFriday = new Date(today);
                nextFriday.setDate(today.getDate() + (diff >= 0 ? diff : diff + 7));
                input.valueAsDate = nextFriday;
            }
            return;
        }

        // Time Logic
        if (this.activeField === 'time') {
            const input = document.getElementById('task-time');
            if (!input) return;
            const normalizeTime = (text) => {
                text = text.toLowerCase();
                if (text.includes('mediodía') || text.includes('medio día')) return '12:00';
                if (text.includes('medianoche')) return '00:00';
                const numbers = text.match(/(\d{1,2})/g);
                if (!numbers || numbers.length === 0) return null;
                let hours = parseInt(numbers[0]);
                let minutes = numbers.length > 1 ? parseInt(numbers[1]) : 0;
                const isPM = text.includes('pm') || text.includes('p.m.') || text.includes('tarde') || text.includes('noche');
                const isAM = text.includes('am') || text.includes('a.m.') || text.includes('mañana');
                if (isPM && hours < 12) hours += 12;
                if (isAM && hours === 12) hours = 0;
                return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
            };
            const parsedTime = normalizeTime(cleanText);
            if (parsedTime) input.value = parsedTime;
            return;
        }

        // Text Fields (Title/Desc)
        if (this.activeField === 'title' || this.activeField === 'desc') {
            const elId = this.activeField === 'title' ? 'task-title' : 'task-desc';
            const el = document.getElementById(elId);
            if (!el) return;

            let currentVal = el.value;
            if (this.lastInterimLength > 0) {
                currentVal = currentVal.substring(0, currentVal.length - this.lastInterimLength);
            }

            // Append
            el.value = currentVal + " " + cleanText; // Add space for continuity
            el.value = el.value.trim(); // Clean edge case

            // Track interim length for replacement
            this.lastInterimLength = cleanText.length + 1;

            // If the event was final, commit it (reset tracker so next speech appends)
            if (finalChunk) {
                this.lastInterimLength = 0;
            }
        }
    }
};
