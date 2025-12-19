"""
Database initialization script
Creates all tables and populates with seed data
"""
from app import create_app, db
from app.models import (
    User, Professional, Service, 
    Comment, ChatMessage,
    Credit, Referral, ReferralEarning, ReferralWithdrawal,
    ChatbotConfig
)
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

def init_database():
    app = create_app()
    with app.app_context():
        # Drop all tables (for fresh start)
        print("🗑️  Dropping existing tables...")
        db.drop_all()
        
        # Create all tables
        print("📊 Creating database tables...")
        db.create_all()
        print("✓ Database tables created successfully")
        
        # Add seed data
        print("🌱 Adding seed data...")
        
        # 1. Create Regular Users
        users = [
            User(
                email='juan.perez@example.com',
                password_hash=generate_password_hash('password123'),
                full_name='Juan Pérez García',
                role='user'
            ),
            User(
                email='maria.lopez@example.com',
                password_hash=generate_password_hash('password123'),
                full_name='María López Martínez',
                role='professional'
            ),
            User(
                email='admin@example.com',
                password_hash=generate_password_hash('admin123'),
                full_name='Administrador',
                role='admin'
            ),
        ]
        db.session.add_all(users)
        db.session.commit()
        
        # 2. Create Professionals (linked to users)
        professionals = [
            Professional(
                user_id=users[0].id,
                specialty='Pensiones IMSS',
                city='Ciudad de México',
                bio='Especialista en pensiones IMSS con más de 10 años de experiencia. He ayudado a cientos de personas a obtener su pensión.',
                rating=4.8,
                total_reviews=45,
                is_active=True
            ),
            Professional(
                user_id=users[1].id,
                specialty='Seguridad Social',
                city='Guadalajara',
                bio='Abogada especializada en seguridad social y trámites IMSS. Atención personalizada y profesional.',
                rating=4.9,
                total_reviews=38,
                is_active=True
            ),
        ]
        db.session.add_all(professionals)
        db.session.commit()
        
        # 3. Create Services for professionals
        services = [
            Service(
                professional_id=professionals[0].id,
                name='Pensión por Cesantía',
                description='Trámite de pensión por cesantía en edad avanzada',
                price=1500.00,
                duration_minutes=60
            ),
            Service(
                professional_id=professionals[0].id,
                name='Pensión por Vejez',
                description='Trámite de pensión por vejez IMSS',
                price=1500.00,
                duration_minutes=60
            ),
            Service(
                professional_id=professionals[1].id,
                name='Alta en IMSS',
                description='Registro de trabajadores en el IMSS',
                price=500.00,
                duration_minutes=30
            ),
            Service(
                professional_id=professionals[1].id,
                name='Incapacidad Temporal',
                description='Gestión de incapacidades temporales',
                price=800.00,
                duration_minutes=45
            ),
        ]
        db.session.add_all(services)
        
        # 4. Create Comments/Reviews
        comments = [
            Comment(
                professional_id=professionals[0].id,
                user_id=users[0].id,
                rating=5,
                content='Excelente servicio, muy profesional y atento. Me ayudó a obtener mi pensión sin problemas.',
                status='approved',
                created_at=datetime.now() - timedelta(days=10)
            ),
            Comment(
                professional_id=professionals[0].id,
                user_id=users[1].id,
                rating=4,
                content='Buen servicio, aunque tardó un poco más de lo esperado.',
                status='approved',
                created_at=datetime.now() - timedelta(days=5)
            ),
            Comment(
                professional_id=professionals[1].id,
                user_id=users[0].id,
                rating=5,
                content='Ana es excelente, muy recomendada. Resolvió todas mis dudas.',
                status='approved',
                created_at=datetime.now() - timedelta(days=3)
            ),
        ]
        db.session.add_all(comments)
        
        # 5. Add credits to professionals for testing chatbot
        credits = [
            Credit(
                professional_id=professionals[0].id,
                amount=100,
                transaction_type='purchase',
                transaction_amount=100,
                payment_method='CLABE',
                payment_status='completed',
                price_mxn=179.00
            ),
            Credit(
                professional_id=professionals[1].id,
                amount=50,
                transaction_type='purchase',
                transaction_amount=50,
                payment_method='CLABE',
                payment_status='completed',
                price_mxn=99.00
            ),
        ]
        db.session.add_all(credits)
        
        # 6. Create chatbot configurations
        chatbot_configs = [
            ChatbotConfig(
                professional_id=professionals[0].id,
                is_active=True,
                welcome_message='¡Hola! Soy el asistente virtual de Carlos. ¿En qué puedo ayudarte con tus trámites IMSS?',
                system_prompt='Eres un asistente experto en pensiones y trámites del IMSS. Ayuda a los usuarios con información clara y precisa.',
                max_tokens=1000,
                temperature=0.7
            ),
            ChatbotConfig(
                professional_id=professionals[1].id,
                is_active=True,
                welcome_message='¡Bienvenido! Soy el asistente de Ana. ¿Cómo puedo ayudarte hoy?',
                system_prompt='Eres un asistente especializado en seguridad social y trámites IMSS. Proporciona asesoría profesional.',
                max_tokens=1000,
                temperature=0.7
            ),
        ]
        db.session.add_all(chatbot_configs)
        
        # Commit all changes
        db.session.commit()
        
        print("✓ Seed data added successfully")
        print(f"✓ Database location: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print("\n📊 Summary:")
        print(f"   - {len(users)} users created")
        print(f"   - {len(professionals)} professionals created")
        print(f"   - {len(services)} services created")
        print(f"   - {len(comments)} reviews created")
        print(f"   - {len(credits)} credit records created")
        print(f"   - {len(chatbot_configs)} chatbot configurations created")
        print("\n🔐 Login Credentials:")
        print("   User 1: juan.perez@example.com / password123")
        print("   User 2 (Professional): maria.lopez@example.com / password123")
        print("\n✅ Database ready! You can now start the application with: python run.py")

if __name__ == '__main__':
    init_database()
