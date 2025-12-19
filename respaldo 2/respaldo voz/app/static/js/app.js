// AsesoriaIMSS.io - Vanilla JavaScript Application
// NO frameworks - Pure ES6+ JavaScript

// Configuration
const CONFIG = {
    apiBaseUrl: '/api',
    language: 'es',
    languages: {
        es: {
            searchPlaceholder: '¿Qué necesitas saber sobre el IMSS?',
            searchButton: 'Buscar',
            loading: 'Cargando...',
            error: 'Error al procesar la solicitud'
        },
        en: {
            searchPlaceholder: 'What do you need to know about IMSS?',
            searchButton: 'Search',
            loading: 'Loading...',
            error: 'Error processing request'
        }
    }
};

// DOM Elements
const elements = {
    searchInput: document.getElementById('searchInput'),
    searchBtn: document.getElementById('searchBtn'),
    resultsSection: document.getElementById('resultsSection'),
    resultsContainer: document.getElementById('resultsContainer'),
    langToggle: document.getElementById('langToggle')
};

// State Management (Simple vanilla JS state)
const state = {
    currentLanguage: 'es',
    isLoading: false,
    lastQuery: ''
};

// Event Listeners
function initEventListeners() {
    // Search button click
    elements.searchBtn.addEventListener('click', handleSearch);

    // Enter key on search input
    elements.searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });

    // Language toggle
    elements.langToggle.addEventListener('click', toggleLanguage);
}

// Search Handler
async function handleSearch() {
    const query = elements.searchInput.value.trim();

    if (!query) {
        showError('Por favor ingresa una consulta');
        return;
    }

    state.lastQuery = query;
    state.isLoading = true;

    showLoading();

    try {
        // API call using Fetch API (vanilla JS)
        const response = await fetch(`${CONFIG.apiBaseUrl}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                language: state.currentLanguage
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Search error:', error);
        showError(CONFIG.languages[state.currentLanguage].error);
    } finally {
        state.isLoading = false;
    }
}

// Display Results
function displayResults(data) {
    elements.resultsSection.classList.remove('hidden');
    elements.resultsSection.classList.add('fade-in');

    // Clear previous results
    elements.resultsContainer.innerHTML = '';

    // Create result HTML (vanilla DOM manipulation)
    const resultHTML = `
        <div class="space-y-4">
            <h4 class="text-xl font-semibold text-gray-800">${escapeHtml(data.title || 'Resultados')}</h4>
            <p class="text-gray-600">${escapeHtml(data.content || 'No se encontraron resultados')}</p>
            ${data.suggestions ? createSuggestionsList(data.suggestions) : ''}
        </div>
    `;

    elements.resultsContainer.innerHTML = resultHTML;

    // Smooth scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create Suggestions List
function createSuggestionsList(suggestions) {
    if (!Array.isArray(suggestions) || suggestions.length === 0) {
        return '';
    }

    const items = suggestions.map(s => `<li class="text-blue-600 hover:underline cursor-pointer">${escapeHtml(s)}</li>`).join('');

    return `
        <div class="mt-4">
            <h5 class="font-semibold text-gray-700 mb-2">Sugerencias relacionadas:</h5>
            <ul class="list-disc list-inside space-y-1">
                ${items}
            </ul>
        </div>
    `;
}

// Show Loading State
function showLoading() {
    elements.resultsSection.classList.remove('hidden');
    elements.resultsContainer.innerHTML = `
        <div class="text-center py-8">
            <div class="spinner"></div>
            <p class="text-gray-600 mt-4">${CONFIG.languages[state.currentLanguage].loading}</p>
        </div>
    `;
}

// Show Error
function showError(message) {
    elements.resultsSection.classList.remove('hidden');
    elements.resultsContainer.innerHTML = `
        <div class="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            <p>${escapeHtml(message)}</p>
        </div>
    `;
}

// Language Toggle
function toggleLanguage() {
    state.currentLanguage = state.currentLanguage === 'es' ? 'en' : 'es';
    elements.langToggle.textContent = state.currentLanguage.toUpperCase();

    // Update UI text
    const lang = CONFIG.languages[state.currentLanguage];
    elements.searchInput.placeholder = lang.searchPlaceholder;
    elements.searchBtn.textContent = lang.searchButton;
}

// Utility: Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('AsesoriaIMSS.io initialized - Vanilla JS');
    initEventListeners();
});

// Export for potential testing (optional)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { handleSearch, toggleLanguage, displayResults };
}
