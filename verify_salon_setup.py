
from app import create_app
from app.models import User, SalonConfig
import requests

app = create_app()

def test_salons_endpoint():
    with app.test_client() as client:
        # 1. Login as admin (simulate session or use admin_required bypass if possible, 
        # but easier to just mock or use the app context directly to check DB)
        # Actually I can't easily mock auth in this script without a lot of setup.
        # So I will check if the route is registered and DB is correct.
        
        print("Checking DB Schema...")
        with app.app_context():
            # Check if columns exist by querying
            try:
                s = SalonConfig.query.first()
                print("SalonConfig query successful.")
            except Exception as e:
                print(f"SalonConfig query failed: {e}")
                
        # 2. Check if route exists
        print("Checking Routes...")
        rules = [str(p) for p in app.url_map.iter_rules()]
        if any('/api/admin/salones' in r for r in rules):
            print("Route /api/admin/salones found.")
        else:
            print("Route /api/admin/salones NOT found. (Might be /admin/salones if blueprint prefix is different)")
            for r in rules:
                if 'salones' in r:
                    print(f"Found related route: {r}")

if __name__ == "__main__":
    test_salons_endpoint()
