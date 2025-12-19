// API Client Wrapper - Centralized API calls with auth and error handling

const API_BASE_URL = '/api';

// Generic API call function
async function apiCall(endpoint, options = {}) {
    const token = getAuthToken();

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` })
        }
    };

    const finalOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers
        }
    };

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, finalOptions);
        const data = await response.json();

        if (!response.ok) {
            // Handle 401 Unauthorized
            if (response.status === 401) {
                clearAuth();
                window.location.href = '/login';
                throw new Error('Sesión expirada. Por favor inicia sesión nuevamente.');
            }

            throw new Error(data.error || `Error ${response.status}`);
        }

        return { success: true, data: data.data || data, message: data.message };
    } catch (error) {
        console.error('API Error:', error);
        return { success: false, error: error.message };
    }
}

// GET request
async function apiGet(endpoint) {
    return apiCall(endpoint, { method: 'GET' });
}

// POST request
async function apiPost(endpoint, body) {
    return apiCall(endpoint, {
        method: 'POST',
        body: JSON.stringify(body)
    });
}

// PUT request
async function apiPut(endpoint, body) {
    return apiCall(endpoint, {
        method: 'PUT',
        body: JSON.stringify(body)
    });
}

// DELETE request
async function apiDelete(endpoint) {
    return apiCall(endpoint, { method: 'DELETE' });
}

// ============================================
// PROFESSIONALS API
// ============================================

async function getProfessionals(filters = {}) {
    const params = new URLSearchParams(filters);
    return apiGet(`/profesionales?${params}`);
}

async function getProfessional(id) {
    return apiGet(`/profesionales/${id}`);
}

async function createProfessional(data) {
    return apiPost('/profesionales', data);
}

async function updateProfessional(id, data) {
    return apiPut(`/profesionales/${id}`, data);
}

async function deleteProfessional(id) {
    return apiDelete(`/profesionales/${id}`);
}

async function getProfessionalServices(id) {
    return apiGet(`/profesionales/${id}/servicios`);
}

async function getProfessionalExperience(id) {
    return apiGet(`/profesionales/${id}/experiencia`);
}

async function getProfessionalCertifications(id) {
    return apiGet(`/profesionales/${id}/certificaciones`);
}

async function getProfessionalComments(id) {
    return apiGet(`/profesionales/${id}/comentarios`);
}

// ============================================
// CHATBOT API
// ============================================

async function sendChatMessage(professionalId, message, sessionId = null) {
    return apiPost(`/chatbot/${professionalId}/mensaje`, {
        message,
        session_id: sessionId
    });
}

async function getChatbotConfig(professionalId) {
    return apiGet(`/chatbot/${professionalId}/config`);
}

async function updateChatbotConfig(professionalId, config) {
    return apiPut(`/chatbot/${professionalId}/config`, config);
}

async function uploadChatbotDocument(professionalId, documentText) {
    return apiPost(`/chatbot/${professionalId}/subir-documento`, {
        document_text: documentText
    });
}

async function getChatHistory(professionalId, sessionId = null) {
    const token = getAuthToken();
    if (token) {
        const params = sessionId ? `?session_id=${sessionId}` : '';
        return apiGet(`/chatbot/${professionalId}/historial${params}`);
    } else {
        // Public access
        if (!sessionId) return { success: true, data: { messages: [] } };
        return apiGet(`/chatbot/${professionalId}/public-history?session_id=${sessionId}`);
    }
}

// ============================================
// COMMENTS API
// ============================================

async function createComment(professionalId, rating, content) {
    return apiPost(`/comentarios/${professionalId}`, { rating, content });
}

async function getComments(professionalId) {
    return apiGet(`/comentarios/${professionalId}`);
}

async function updateComment(id, data) {
    return apiPut(`/comentarios/${id}`, data);
}

async function deleteComment(id) {
    return apiDelete(`/comentarios/${id}`);
}

// ============================================
// CREDITS API
// ============================================

async function purchaseCredits(amount, paymentMethod) {
    return apiPost('/creditos/comprar', {
        amount,
        payment_method: paymentMethod
    });
}

async function getCredits(professionalId) {
    return apiGet(`/creditos/${professionalId}`);
}

// ============================================
// REFERRALS API
// ============================================

async function generateReferralLink() {
    return apiPost('/referrals/generar-link', {});
}

async function getActiveReferrals(professionalId) {
    return apiGet(`/referrals/${professionalId}/activos`);
}

async function getReferralEarnings(professionalId) {
    return apiGet(`/referrals/${professionalId}/ganancias`);
}

async function requestWithdrawal(amount, method, clabeAccount = null) {
    return apiPost('/referrals/solicitar-retiro', {
        amount_mxn: amount,
        withdrawal_method: method,
        clabe_account: clabeAccount
    });
}

// ============================================
// ADMIN API
// ============================================

async function getAdminDashboard() {
    return apiGet('/admin/dashboard');
}

async function getPendingComments(page = 1) {
    return apiGet(`/admin/comentarios-pendientes?page=${page}`);
}

async function updateCommentStatus(id, status) {
    return apiPut(`/admin/comentarios/${id}/estado`, { status });
}

async function sendEmailMarketing(subject, content, target) {
    return apiPost('/admin/email-marketing', {
        subject,
        content,
        target
    });
}

async function confirmPayment(transactionId) {
    return apiPost('/creditos/confirmar-pago', {
        transaction_id: transactionId
    });
}
