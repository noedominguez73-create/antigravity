from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    admin = User.query.filter_by(email='admin@asesoriaimss.io').first()
    if admin:
        print(f"Found admin user: {admin.email}")
        admin.password_hash = generate_password_hash('Admin123!')
        db.session.commit()
        print("Password updated successfully to 'Admin123!'")
    else:
        print("Admin user not found. Creating one...")
        admin = User(
            email='admin@asesoriaimss.io',
            full_name='Administrador Sistema',
            role='admin',
            password_hash=generate_password_hash('Admin123!')
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created with password 'Admin123!'")
