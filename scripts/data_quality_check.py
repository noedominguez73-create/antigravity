import sys
import os
import pandas as pd
from sqlalchemy import text

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, Professional

def run_checks():
    app = create_app()
    
    with app.app_context():
        print("[INFO] Iniciando Auditoria de Calidad de Datos (Data Quality Check)...\n")
        
        # Use pandas to read from the database engine
        try:
            # 1. Load DataFrames
            print("[INFO] Cargando datos...")
            df_users = pd.read_sql(text("SELECT * FROM users"), db.engine)
            df_profs = pd.read_sql(text("SELECT * FROM professionals"), db.engine)
            
            print(f"   - Usuarios cargados: {len(df_users)}")
            print(f"   - Profesionales cargados: {len(df_profs)}")
            print("-" * 50)

            # ---------------------------------------------------------
            # CHECK 1: Completitud de Perfiles
            # ---------------------------------------------------------
            print("\n[CHECK 1] Completitud de Perfiles Profesionales")
            # Check for null or empty bio/specialty
            incomplete_profs = df_profs[
                (df_profs['bio'].isnull()) | (df_profs['bio'] == '') |
                (df_profs['specialty'].isnull()) | (df_profs['specialty'] == '')
            ]
            
            if len(incomplete_profs) > 0:
                print(f"   [ALERTA] {len(incomplete_profs)} profesionales con perfil incompleto.")
                print(incomplete_profs[['id', 'user_id', 'specialty']].head())
            else:
                print("   [OK] Todos los perfiles tienen Bio y Especialidad.")

            # ---------------------------------------------------------
            # CHECK 2: Unicidad de Email (Normalizado)
            # ---------------------------------------------------------
            print("\n[CHECK 2] Unicidad de Email (Normalizado)")
            if not df_users.empty:
                df_users['email_norm'] = df_users['email'].str.lower().str.strip()
                duplicates = df_users[df_users.duplicated('email_norm', keep=False)]
                
                if len(duplicates) > 0:
                    print(f"   [ALERTA] {len(duplicates)} usuarios con email duplicado (case-insensitive).")
                    print(duplicates[['id', 'email', 'email_norm']].sort_values('email_norm'))
                else:
                    print("   [OK] No se detectaron emails duplicados.")
            else:
                print("   [INFO] Tabla de usuarios vacia.")

            # ---------------------------------------------------------
            # CHECK 3: Consistencia de Roles
            # ---------------------------------------------------------
            print("\n[CHECK 3] Consistencia de Roles (User vs Professional)")
            if not df_profs.empty and not df_users.empty:
                merged = df_profs.merge(df_users, left_on='user_id', right_on='id', suffixes=('_prof', '_user'))
                role_mismatch = merged[merged['role'] != 'professional']
                
                if len(role_mismatch) > 0:
                    print(f"   [ALERTA] {len(role_mismatch)} profesionales tienen rol incorrecto en tabla users.")
                    print(role_mismatch[['id_user', 'email', 'role']])
                else:
                    print("   [OK] Todos los profesionales tienen el rol 'professional' asignado.")

            # ---------------------------------------------------------
            # CHECK 4: Saldos Negativos (Critical)
            # ---------------------------------------------------------
            print("\n[CHECK 4] Auditoria de Saldos (Negativos)")
            # SQL logic translated to Pandas for flexibility or raw SQL
            sql_balance = text("""
                WITH CreditsAdded AS (
                    SELECT professional_id, SUM(transaction_amount) as total_added
                    FROM credits
                    WHERE payment_status = 'confirmed'
                    GROUP BY professional_id
                ),
                CreditsSpent AS (
                    SELECT professional_id, SUM(credits_used) as total_spent
                    FROM chat_messages
                    GROUP BY professional_id
                )
                SELECT 
                    p.id, 
                    p.user_id, 
                    COALESCE(ca.total_added, 0) as added,
                    COALESCE(cs.total_spent, 0) as spent,
                    (COALESCE(ca.total_added, 0) - COALESCE(cs.total_spent, 0)) as real_balance
                FROM professionals p
                LEFT JOIN CreditsAdded ca ON p.id = ca.professional_id
                LEFT JOIN CreditsSpent cs ON p.id = cs.professional_id
                WHERE (COALESCE(ca.total_added, 0) - COALESCE(cs.total_spent, 0)) < 0
            """)
            
            negative_balances = pd.read_sql(sql_balance, db.engine)
            
            if len(negative_balances) > 0:
                print(f"   [CRITICO] {len(negative_balances)} profesionales con SALDO NEGATIVO.")
                print(negative_balances)
            else:
                print("   [OK] No se detectaron saldos negativos.")

            print("\n" + "="*50)
            print("[FIN] Auditoria Finalizada.")
            print("="*50)

        except Exception as e:
            print(f"\n❌ Error ejecutando auditoría: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_checks()
