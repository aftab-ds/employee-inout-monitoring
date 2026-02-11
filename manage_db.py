
import argparse
from src.database import Database
import time

def list_persons(db):
    persons = db.get_all_embeddings()
    print(f"\n{'ID':<5} {'Name':<20} {'Status':<10} {'Entry Time'}")
    print("-" * 60)
    for p in persons:
        status_str = "IN" if p['status'] == 1 else "OUT"
        entry_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p['entry_time']))
        print(f"{p['id']:<5} {p['name']:<20} {status_str:<10} {entry_time_str}")
    print("-" * 60)

def delete_person(db, person_id):
    try:
        db.cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        db.conn.commit()
        print(f"Deleted Person ID {person_id}.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage Office Productivity DB")
    parser.add_argument("--list", action="store_true", help="List all persons")
    parser.add_argument("--delete", type=int, help="Delete person by ID")
    parser.add_argument("--cleanup", action="store_true", help="Delete all persons")
    
    args = parser.parse_args()
    db = Database()
    
    if args.delete:
        delete_person(db, args.delete)
        
    if args.cleanup:
        confirm = input("Are you sure you want to delete ALL records? (y/n): ")
        if confirm.lower() == 'y':
            db.cursor.execute("DELETE FROM persons")
            db.conn.commit()
            print("Database cleared.")

    # Always list at the end if specific action wasn't just a deletion that might make list empty/confusing? 
    # No, always list is good feedback.
    list_persons(db)
    
    db.close()

if __name__ == "__main__":
    main()
