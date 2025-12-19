from app import create_app, db
from app.models import ChatbotConfig, Professional

app = create_app()

with app.app_context():
    prof_id = 3
    prof = Professional.query.get(prof_id)
    if not prof:
        print(f"Professional {prof_id} not found")
    else:
        print(f"Professional ID: {prof.id}")
        if prof.user:
            print(f"User: {prof.user.full_name}")
        config = ChatbotConfig.query.filter_by(professional_id=prof_id).first()
        if config:
            print(f"Config found: ID={config.id}, Active={config.is_active}")
        else:
            print("No ChatbotConfig found for this professional")
