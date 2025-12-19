# 🚀 Guía de Despliegue en Hostinger (Python/Flask)

Esta guía te ayudará a subir tu proyecto **Mirror IA** a Hostinger paso a paso.

## 1. Preparación de Archivos (Local)
Asegúrate de tener estos archivos listos en tu computadora:
- [x] `requirements.txt` (Ya está actualizado en tu proyecto).
- [x] `passenger_wsgi.py` (Acabo de crearlo, es vital para que arranque).
- [x] Carpeta del proyecto completa (sin `venv` ni `__pycache__`).

## 2. Configuración en Hostinger (cPanel / hPanel)

1.  **Entra a tu Hosting** > **Sitios Web** > **Administrar**.
2.  Busca la sección **Avanzado** y selecciona **Python App** (o "Setup Python App").
3.  **Crear Nueva Aplicación**:
    *   **Python Version**: Selecciona la recomendada (3.9 o superior).
    *   **App Directory**: Escribe el nombre de la carpeta, ej: `asesoriaimss_io`.
    *   **Application domain**: Selecciona tu dominio/subdominio.
    *   **Click en CREAR**.

## 3. Subir Archivos

1.  Entra al **Administrador de Archivos** (File Manager).
2.  Navega a la carpeta que acabas de crear (ej: `/home/u12345/virtualenv/asesoriaimss_io/...` NO, busca la carpeta raíz de la app, usualmente en el home o public_html).
3.  **Sube todo tu código** ahí.
    *   ⚠️ **IMPORTANTE:** No subas tu carpeta `venv` local. El servidor creará la suya.
    *   Asegúrate de que `passenger_wsgi.py` esté junto a `run.py`.
    *   Sube tu archivo `.env` con las claves (API KEYS).

## 4. Instalar Dependencias

1.  Vuelve a la pantalla de **Python App** en Hostinger.
2.  Verás que detectó tu aplicación.
3.  Busca el campo **Configuration file** y escribe: `requirements.txt`.
4.  Haz click en el botón **Run Pip Install** (o "Instalar dependencias").
    *   *Esto tardará unos minutos mientras descarga Flask, Edge-TTS, etc.*
    *   Si falla, revisa los logs. A veces `psycopg2-binary` da problemas en Linux, si es así, cámbialo por `psycopg2` en el txt.

## 5. Permisos de Carpetas (Crucial para Voz e Imágenes)

Para que la IA pueda guardar fotos y audios, necesitas dar permisos de escritura:
1.  En el Administrador de Archivos.
2.  Ve a `app/static`.
3.  Click derecho en la carpeta `uploads` -> **Permissions (Permisos)**.
4.  Asegúrate de que tenga **755** (o 777 si da problemas, pero 755 es más seguro).
5.  Repite para `app/routes` (si ahí guardas el json de config) o mueve el json a `app/static` si prefieres.

## 6. Reiniciar

1.  En la pantalla de Python App, dale al botón **RESTART**.
2.  Visita tu web. ¡Debería funcionar!

## 🆘 Solución de Problemas Comunes

*   **Error 500 / Pantalla Blanca:**
    *   Revisa el archivo `passenger_wsgi.py`.
    *   Asegúrate de que `application` está siendo importado correctamente.
*   **No se escucha la voz:**
    *   Seguramente falta instalar `edge-tts` o falta `ffmpeg` en el servidor (raro en hostinger, suelen tenerlo).
    *   Verifica los permisos de la carpeta `uploads/audio`.

---
*Generado por Asesora IA Dev Team*
