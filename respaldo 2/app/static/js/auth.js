// Authentication Utilities - JWT Management and User Session

const AUTH_TOKEN_KEY = 'asesoriaimss_token';
const AUTH_USER_KEY = 'asesoriaimss_user';

// Get current user from localStorage
function getCurrentUser() {
    const userStr = localStorage.getItem(AUTH_USER_KEY);
    return userStr ? JSON.parse(userStr) : null;
}

// Get auth token
function getAuthToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY);
}

// Check if user is authenticated
function isAuthenticated() {
    return !!getAuthToken();
}

// Check if user has specific role
function hasRole(role) {
    const user = getCurrentUser();
    return user && user.role === role;
}

// Save authentication data
function saveAuth(token, user) {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

// Clear authentication data
function clearAuth() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
}

// Login function
async function login(email, password) {
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error al iniciar sesión');
        }

        if (data.success && data.data.token) {
            saveAuth(data.data.token, data.data.user);
            return { success: true, user: data.data.user };
        }

        throw new Error('Respuesta inválida del servidor');
    } catch (error) {
        console.error('Login error:', error);
        return { success: false, error: error.message };
    }
}

// Register function
async function register(userData) {
    try {
        const response = await fetch('/api/auth/registro', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(userData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error al registrar usuario');
        }

        if (data.success && data.data.token) {
            saveAuth(data.data.token, data.data.user);
            return { success: true, user: data.data.user };
        }

        throw new Error('Respuesta inválida del servidor');
    } catch (error) {
        console.error('Register error:', error);
        return { success: false, error: error.message };
    }
}

// Logout function
function logout() {
    clearAuth();
    window.location.href = '/';
}

// OAuth login (Google)
async function loginWithGoogle(googleData) {
    try {
        const response = await fetch('/api/auth/google-login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(googleData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error al iniciar sesión con Google');
        }

        if (data.success && data.data.token) {
            saveAuth(data.data.token, data.data.user);
            return { success: true, user: data.data.user };
        }

        throw new Error('Respuesta inválida del servidor');
    } catch (error) {
        console.error('Google login error:', error);
        return { success: false, error: error.message };
    }
}

// OAuth login (Facebook)
async function loginWithFacebook(facebookData) {
    try {
        const response = await fetch('/api/auth/facebook-login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(facebookData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Error al iniciar sesión con Facebook');
        }

        if (data.success && data.data.token) {
            saveAuth(data.data.token, data.data.user);
            return { success: true, user: data.data.user };
        }

        throw new Error('Respuesta inválida del servidor');
    } catch (error) {
        console.error('Facebook login error:', error);
        return { success: false, error: error.message };
    }
}

// Redirect based on user role
function redirectToDashboard() {
    const user = getCurrentUser();
    if (!user) {
        window.location.href = '/login';
        return;
    }

    switch (user.role) {
        case 'admin':
            window.location.href = '/admin';
            break;
        case 'professional':
            window.location.href = '/dashboard-profesional';
            break;
        case 'user':
            window.location.href = '/perfil-usuario';
            break;
        case 'tienda':
            window.location.href = '/tienda';
            break;
        default:
            window.location.href = '/';
    }
}

// Require authentication (call on protected pages)
function requireAuth() {
    if (!isAuthenticated()) {
        window.location.href = '/login?redirect=' + encodeURIComponent(window.location.pathname);
        return false;
    }
    return true;
}

// Require specific role
function requireRole(role) {
    if (!requireAuth()) return false;

    if (!hasRole(role)) {
        showToast('No tienes permiso para acceder a esta página', 'error');
        redirectToDashboard();
        return false;
    }
    return true;
}

// Update UI based on auth state
function updateAuthUI() {
    const user = getCurrentUser();
    const authButtons = document.getElementById('authButtons');
    const userMenu = document.getElementById('userMenu');

    if (user && authButtons && userMenu) {
        authButtons.classList.add('hidden');
        userMenu.classList.remove('hidden');

        const userName = document.getElementById('userName');
        if (userName) {
            userName.textContent = user.full_name || user.email;
        }
    } else if (authButtons && userMenu) {
        authButtons.classList.remove('hidden');
        userMenu.classList.add('hidden');
    }
}

// Initialize auth on page load
document.addEventListener('DOMContentLoaded', () => {
    updateAuthUI();
});
