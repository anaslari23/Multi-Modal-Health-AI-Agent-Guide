import csv
import sys
import os

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, engine, Base
from app.db_models import Medicine

def get_max_id(db):
    """Get the maximum ID currently in the database."""
    max_id = db.query(Medicine.id).order_by(Medicine.id.desc()).first()
    return max_id[0] if max_id else 0

def import_medicines_from_file(db, csv_path, start_id):
    """Import medicines from a CSV file, using sequential IDs starting from start_id."""
    print(f"Importing from {csv_path}...")
    print(f"  Starting from ID: {start_id}")
    
    count = 0
    batch = []
    current_id = start_id
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Handle price conversion (column might be 'price' or 'price(₹)')
            price_col = 'price' if 'price' in row else 'price(₹)'
            try:
                price_str = row.get(price_col, '').strip()
                price = float(price_str) if price_str else None
            except (ValueError, KeyError):
                price = None

            medicine = Medicine(
                id=current_id,
                name=row.get('name', ''),
                price=price,
                is_discontinued=row.get('Is_discontinued', 'FALSE'),
                manufacturer_name=row.get('manufacturer_name'),
                type=row.get('type'),
                pack_size_label=row.get('pack_size_label'),
                short_composition1=row.get('short_composition1'),
                short_composition2=row.get('short_composition2'),
                salt_composition=row.get('salt_composition'),
                medicine_desc=row.get('medicine_desc'),
                side_effects=row.get('side_effects'),
                drug_interactions=row.get('drug_interactions')
            )
            batch.append(medicine)
            count += 1
            current_id += 1
            
            if len(batch) >= 1000:
                db.add_all(batch)
                db.commit()
                print(f"  Imported {count} medicines...")
                batch = []
        
        if batch:
            db.add_all(batch)
            db.commit()
            
    print(f"  Completed: {count} medicines imported from this file.")
    return count

def import_second_dataset():
    """Import only the second dataset (A_Z) without clearing existing data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    try:
        # Get the max ID from existing data
        max_id = get_max_id(db)
        print(f"Found {db.query(Medicine).count()} existing medicines.")
        print(f"Max ID in database: {max_id}")
        
        # Import from second dataset starting after the max ID
        csv2 = "../datasets/medicine/A_Z_medicines_dataset_of_India.csv"
        if os.path.exists(csv2):
            count = import_medicines_from_file(db, csv2, start_id=max_id + 1)
            total = db.query(Medicine).count()
            print(f"\n✅ Successfully imported {count} new medicines!")
            print(f"📊 Total medicines in database: {total}")
        else:
            print(f"❌ File not found: {csv2}")
        
    except Exception as e:
        print(f"❌ Error importing medicines: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    import_second_dataset()
