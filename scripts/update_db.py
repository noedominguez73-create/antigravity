from app import create_app, db
from app.models import Store, StoreType

app = create_app()

with app.app_context():
    try:
        print(f"DEBUG: Using Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print(f"DEBUG: Engine URL: {db.engine.url}")
        
        # Create StoreType table
        # Create StoreType table
        if not db.engine.dialect.has_table(db.engine, "store_types"):
            print("Creating table 'store_types'...")
            StoreType.__table__.create(db.engine)
        else:
            print("Table 'store_types' already exists.")

        # Create Store table
        if not db.engine.dialect.has_table(db.engine, "stores"):
            print("Creating table 'stores'...")
            Store.__table__.create(db.engine)
        else:
            print("Table 'stores' already exists.")
            
        print("Database update completed.")
    except Exception as e:
        print(f"Error updating database: {e}")
