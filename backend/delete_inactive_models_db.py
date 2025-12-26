"""
Direct database script to delete inactive models
Run inside Docker: docker exec jarvistrade_backend python delete_inactive_models_db.py
"""
from app.db.database import SessionLocal
from app.db.models import Model
import os
import shutil

db = SessionLocal()

# Get all models
models = db.query(Model).all()

print("="*80)
print("CURRENT MODELS")
print("="*80)
print(f"{'Status':<10} | {'Name':<40} | {'ID':<10}")
print("="*80)

active_count = 0
inactive_count = 0

for m in models:
    status = "ACTIVE" if m.is_active else "inactive"
    if m.is_active:
        active_count += 1
    else:
        inactive_count += 1
    print(f"{status:<10} | {m.name[:40]:<40} | {str(m.id)[:8]}...")

print("="*80)
print(f"Total: {len(models)} models ({active_count} active, {inactive_count} inactive)")
print("="*80)

# Delete inactive models
if inactive_count == 0:
    print("\n✓ No inactive models to delete!")
else:
    print(f"\nDeleting {inactive_count} inactive model(s)...\n")
    
    deleted = 0
    for m in models:
        if not m.is_active:
            model_id = str(m.id)
            model_name = m.name
            model_path = m.model_path
            
            print(f"Deleting: {model_name[:40]:<40} ", end="")
            
            try:
                # Delete model files if they exist
                if model_path and os.path.exists(model_path):
                    shutil.rmtree(model_path)
                
                # Delete from database
                db.delete(m)
                db.commit()
                
                print("✓ Deleted")
                deleted += 1
            except Exception as e:
                print(f"✗ Error: {str(e)}")
                db.rollback()
    
    print(f"\n✓ Successfully deleted {deleted} model(s)")

db.close()
