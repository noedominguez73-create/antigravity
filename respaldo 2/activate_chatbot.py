from app import create_app, db
from app.models import ChatbotConfig

app = create_app()

with app.app_context():
    prof_id = 3
    config = ChatbotConfig.query.filter_by(professional_id=prof_id).first()
    if config:
        config.is_active = True
        db.session.commit()
        print(f"Chatbot for professional {prof_id} activated successfully.")
    else:
        print(f"No config found for professional {prof_id}")
