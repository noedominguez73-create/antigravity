
import logging
# Disable logging before importing app
logging.disable(logging.CRITICAL)

from app import create_app, db
from app.models import User

# Further suppress standard logging if possible
import os
os.environ["WERKZEUG_RUN_MAIN"] = "true"  # Trick to suppress some flask output? No, opposite.

app = create_app()

with app.app_context():
    users = User.query.all()
    print("--- USERS START ---")
    print(f"Total users: {len(users)}")
    for u in users:
        print(f"ID: {u.id} | Email: {u.email} | Role: {u.role} | FullName: {u.full_name}")
    print("--- USERS END ---")
