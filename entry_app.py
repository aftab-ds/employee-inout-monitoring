
import cv2
import time
import numpy as np
from src.detector import PersonDetector
from src.reid import ReIdentifier
from src.database import Database

import argparse

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Camera Source (0, 1, or video file)")
    args = parser.parse_args()
    
    SOURCE = int(args.source) if args.source.isdigit() else args.source
    MATCH_THRESHOLD = 0.6
    
    detector = PersonDetector()
    reid = ReIdentifier()
    db = Database() 
    
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source {SOURCE}.")
        return
    
    cv2.namedWindow("Entry Camera", cv2.WINDOW_NORMAL)
    
    print("Entry Camera Started. Press 'q' to quit. Press 'r' to register STRANGERS.")
    
    # Adjusted threshold (0.65 is more balanced for MobileNetV3)
    MATCH_THRESHOLD = 0.65
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (640, 480))
        
        tracks = detector.track(frame_resized)
        
        # Get all known people once per loop (or cache it)
        known_persons = db.get_all_embeddings()
        
        # Draw tracks & Identify
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track
            bbox = [x1, y1, x2, y2]
            
            # Extract feature
            current_feature = reid.extract_features(frame_resized, bbox)
            
            best_match_name = "Stranger"
            max_sim = 0.0
            best_match_id = None
            is_known = False
            
            if current_feature is not None and known_persons:
                for person in known_persons:
                    # Compare against ALL embeddings for this person
                    for emb in person['embeddings']:
                        sim = reid.compute_similarity(current_feature, emb)
                        if sim > max_sim:
                            max_sim = sim
                            best_match_id = person['id']
                            best_match_name = person['name']
            
            if max_sim > MATCH_THRESHOLD:
                is_known = True
                color = (0, 255, 0) # Green for known
                label = f"{best_match_name} ({max_sim:.2f})"
                
                # Auto-mark IN if known
                # Check current status and update if needed
                # Also throttle updates to avoid DB spam (e.g. only if last update > 5s ago?)
                # For simplicity: Update if status is 0 (OUT) OR if entry_time is old (> 60s ago)
                current_time = time.time()
                
                # We need to find the person dict in known_persons to check status
                matched_person = next((p for p in known_persons if p['id'] == best_match_id), None)
                
                if matched_person:
                    if matched_person['status'] == 0 or (current_time - matched_person['entry_time'] > 60):
                        db.update_status(best_match_id, 1)
                        print(f"Welcome back, {best_match_name}! Marked IN.")
                        
                        # Update local cache to prevent immediate re-update
                        matched_person['status'] = 1
                        matched_person['entry_time'] = current_time
            else:
                is_known = False
                color = (0, 0, 255) # Red for stranger
                label = f"Stranger ({max_sim:.2f})"
            
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame_resized, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not is_known:
                cv2.putText(frame_resized, "Press 'r' to Register", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Entry Camera", frame_resized)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
             # REGISTRATION MODE
             # Filter tracks to find STRANGERS only
             found_stranger = False
             
             candidates = []
             for track in tracks:
                 bbox = track[:4]
                 feat = reid.extract_features(frame_resized, bbox)
                 
                 # Check against DB
                 sim_max = 0
                 if feat is not None:
                     for p in known_persons:
                         s = reid.compute_similarity(feat, p['embedding'])
                         if s > sim_max: sim_max = s
                 
                 if sim_max < MATCH_THRESHOLD:
                     area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                     candidates.append((area, feat))
            
             candidates.sort(key=lambda x: x[0], reverse=True)
             
             if candidates:
                 target_feat = candidates[0][1]
                 # Pause display
                 cv2.putText(frame_resized, "PAUSED FOR REGISTRATION", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                 cv2.imshow("Entry Camera", frame_resized)
                 cv2.waitKey(1)
                 
                 print("\n--- NEW REGISTRATION ---")
                 name = input("Enter Person Name: ").strip()
                 
                 # Check for duplicate name
                 name_exists = any(p['name'].lower() == name.lower() for p in known_persons)
                 
                 if name and not name_exists:
                     pid = db.add_person(name, target_feat)
                     print(f"Registered {name} (ID: {pid}).")
                 elif name_exists:
                     # Add to existing person
                     pid = next(p['id'] for p in known_persons if p['name'].lower() == name.lower())
                     db.add_embedding(pid, target_feat)
                     print(f"Added new embedding for existing person {name} (ID: {pid}).")
                 else:
                     print("Cancelled.")
             else:
                 print("No stranger detected to register (or they are already known).")

    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    main()
