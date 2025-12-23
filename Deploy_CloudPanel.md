# 🚀 Guía de Despliegue en Hostinger VPS con CloudPanel

Esta guía está diseñada para tu configuración actual: **Ubuntu 24.04 + CloudPanel**.

## 1. Prerrequisitos en CloudPanel

1.  **Entra a tu CloudPanel** (https://tu-ip:8443).
2.  Ve a **Admin Area** > **Users** y crea un usuario si no quieres usar admin.
3.  Ve a **Sites** > **Add Site** > **Create Python Site**.
    *   **Domain Name**: tu-dominio.com (o la IP si no tienes dominio aún).
    *   **Python Version**: 3.10
    *   **App Port**: 5000 (El puerto donde correrá Flask).

## 2. Subir tu Proyecto

1.  **Conexión SFTP**: Usa FileZilla o el File Manager de CloudPanel.
2.  Ve a `/home/nombre-usuario/htdocs/tu-dominio.com/`.
3.  Borra los archivos de ejemplo.
4.  **Sube tus archivos**:
    *   Carpeta `app/`
    *   `run.py`
    *   `requirements.txt`
    *   `passenger_wsgi.py` (Opcional en CloudPanel, pero no estorba).
    *   `.env`

## 3. Configurar Entorno y Dependencias

Si CloudPanel no instala las dependencias automáticamente, hazlo vía SSH:

Connèctate: `ssh usuario@tu-ip`

```bash
cd /home/nombre-usuario/htdocs/tu-dominio.com/

# 1. Crear entorno virtual (si no existe)
python3 -m venv venv

# 2. Activar
source venv/bin/activate

# 3. Instalar librerías
pip install -r requirements.txt
```

## 4. Configurar Gunicorn en CloudPanel

En la configuración del sitio en CloudPanel, asegúrate de que el **Run Command** o script apunte a Gunicorn:

```bash
/home/nombre-usuario/htdocs/tu-dominio.com/venv/bin/gunicorn -w 3 -b 0.0.0.0:5000 run:app
```

## 5. Permisos (Importante para SQLite y Uploads)

Ejecuta esto en SSH para asegurar que la app pueda escribir en la DB y guardar audios:

```bash
# Reemplaza 'nombre-usuario' por tu usuario de CloudPanel
chown -R nombre-usuario:nombre-usuario /home/nombre-usuario/htdocs/tu-dominio.com/
chmod -R 755 /home/nombre-usuario/htdocs/tu-dominio.com/
```

## 6. Verificación

Visita tu sitio. Si ves un error 502, revisa los logs en CloudPanel.
