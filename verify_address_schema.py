
from app import create_app
from app.models import SalonConfig
import requests

app = create_app()

def verify_address_columns():
    with app.app_context():
        # Check columns existed in model (basic check)
        columns = SalonConfig.__table__.columns.keys()
        expected = ['city', 'state', 'country', 'address', 'salon_name']
        missing = [c for c in expected if c not in columns]
        
        if missing:
            print(f"FAILED: Missing columns in SalonConfig model: {missing}")
        else:
            print("SUCCESS: All address columns present in SalonConfig model.")

if __name__ == "__main__":
    verify_address_columns()
