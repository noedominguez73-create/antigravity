from flask import jsonify, current_app
import logging

def error_response(message, status_code=400, internal_error=None):
    """
    Genera una respuesta de error JSON estandarizada.
    Sanitiza errores 500 para no exponer detalles técnicos.
    
    Args:
        message: El mensaje de error o excepción
        status_code: Código HTTP (default 400)
        internal_error: Excepción original (opcional) para logging
        
    Returns:
        Response object con JSON {'success': False, 'error': message}
    """
    # Convertir mensaje a string por si es una excepción
    error_msg = str(message)
    
    # Si es error 500, loguear y ocultar mensaje real si es técnico
    if status_code == 500:
        # Loguear el error real con nivel ERROR
        log_msg = f"INTERNAL SERVER ERROR: {error_msg}"
        if internal_error:
            log_msg += f" | Exception: {str(internal_error)}"
        current_app.logger.error(log_msg)
        
        # Ocultar detalles técnicos si parece una excepción de código/BD
        # Palabras clave comunes en excepciones técnicas
        technical_terms = ["Error:", "Exception", "line", "Traceback", "syntax", "database", "SQL"]
        
        if any(term in error_msg for term in technical_terms):
            error_msg = "Ocurrió un error interno en el servidor. Por favor contacte a soporte."
            
    else:
        # Para otros errores (400, 404, etc), loguear como WARNING
        current_app.logger.warning(f"Client Error ({status_code}): {error_msg}")

    response = jsonify({
        'success': False,
        'error': error_msg
    })
    response.status_code = status_code
    return response
