from flask import Blueprint, render_template, request, redirect, url_for
import jwt
import os

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/registro')
def registro():
    return render_template('registro.html')

@main_bp.route('/tienda')
def tienda():
    return render_template('tienda.html')

@main_bp.route('/login')
def login():
    return render_template('login.html')

@main_bp.route('/profesional')
def profesional():
    return render_template('profesional.html')

@main_bp.route('/dashboard-profesional')
def dashboard_profesional():
    return render_template('dashboard-profesional.html')

@main_bp.route('/creditos')
def creditos():
    return render_template('creditos.html')

@main_bp.route('/perfil-usuario')
def perfil_usuario():
    """Render the user profile page."""
    return render_template('perfil-usuario.html')

@main_bp.route('/referrals')
def referrals():
    return render_template('referrals.html')

@main_bp.route('/chatbot-config')
def chatbot_config():
    return render_template('chatbot-config.html')

@main_bp.route('/admin/login')
def admin_login_page():
    """Show admin login form"""
    return render_template('admin-login.html')

@main_bp.route('/admin')
def admin():
    """Admin dashboard - checks authentication via client-side token"""
    # The authentication is handled client-side via JWT in localStorage
    # If user is not authenticated, the admin.html page will redirect to /admin/login
    # If user is authenticated as admin, it will show the dashboard
    return render_template('admin.html')

@main_bp.route('/chat-history')
def chat_history():
    return render_template('chat-history.html')

@main_bp.route('/mirror')
def mirror():
    return render_template('mirror.html')

@main_bp.route('/fotografia')
def fotografia():
    return render_template('fotografia.html')

@main_bp.route('/closet')
def closet():
    return render_template('closet.html')

@main_bp.route('/cambio_de_imagen')
def cambio_de_imagen():
    return render_template('cambio_de_imagen.html')

@main_bp.route('/mis-finanzas')
def mis_finanzas():
    return render_template('mis_finanzas_dashboard.html')

@main_bp.route('/mis-finanzas/ingresos')
def mis_finanzas_ingresos():
    return render_template('mis_finanzas_ingresos.html')

@main_bp.route('/mis-finanzas/facturas')
def mis_finanzas_facturas():
    return render_template('mis_finanzas_facturas.html')

@main_bp.route('/mis-finanzas/pagos')
def mis_finanzas_pagos():
    return render_template('mis_finanzas_pagos.html')

@main_bp.route('/mis-finanzas/pendientes')
def mis_finanzas_pendientes():
    return render_template('mis_finanzas_pendientes.html')

@main_bp.route('/mis-finanzas/reportes')
def mis_finanzas_reportes():
    return render_template('mis_finanzas_reportes.html')
