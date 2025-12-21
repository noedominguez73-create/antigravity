
from app import create_app
from app.models import User, Professional

app = create_app()

with app.app_context():
    # Simulate the Dashboard logic
    total_users = User.query.filter(User.role != 'admin').count()
    regular_users = User.query.filter_by(role='user').count()
    professionals = Professional.query.filter_by(is_active=True).count()
    
    print(f"Final Dashboard Stats:")
    print(f"Total Users (Active count displayed): {total_users}")
    print(f"Regular Users (Internal): {regular_users}")
    print(f"Professionals (Card 2): {professionals}")
    
    if total_users >= regular_users + professionals:
         print("SUCCESS: Total users reflects the sum (or more) of subgroups.")
    else:
         print(f"WARNING: Total {total_users} is less than sum {regular_users + professionals}?")
