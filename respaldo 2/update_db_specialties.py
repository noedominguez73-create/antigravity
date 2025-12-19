from app import create_app, db
from app.models import Specialty

app = create_app()

with app.app_context():
    # Create the table
    try:
        Specialty.__table__.create(db.engine)
        print("Table 'specialties' created.")
    except Exception as e:
        print(f"Table might already exist: {e}")

    # Seed defaults
    defaults = [
        "Abogado", "Contador", "Arquitecto", "Médico", 
        "Psicólogo", "Nutriólogo", "Dentista", "Ingeniero", 
        "Diseñador", "Desarrollador"
    ]

    for name in defaults:
        if not Specialty.query.filter_by(name=name).first():
            db.session.add(Specialty(name=name))
            print(f"Added: {name}")
    
    db.session.commit()
    print("Seeding complete.")
