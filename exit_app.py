
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
    
    # Convert to int if digit
    SOURCE = int(args.source) if args.source.isdigit() else args.source
    # Adjusted threshold (0.65 is more balanced for MobileNetV3)
    MATCH_THRESHOLD = 0.65 
    
    detector = PersonDetector()
    reid = ReIdentifier()
    db = Database()
    
    print(f"Attempting to open source: {SOURCE}")
    cap = cv2.VideoCapture(SOURCE)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video source {SOURCE}.")
        print("If using a single webcam, make sure 'entry_app.py' is closed before running 'exit_app.py'.")
        return
    
    cv2.namedWindow("Exit Camera", cv2.WINDOW_NORMAL)
    
    print("Exit Camera Started. Press 'q' to quit.")
    
    # Store recently logged exits to avoid spamming
    # {person_id: last_log_time}
    recent_exits = {}
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (640, 480))
        
        tracks = detector.track(frame_resized)
        
        # Get persons who are currently IN (1)
        # Optimization: Only fetch status=1?
        # For robustness, fetch all, but prioritize status=1
        all_persons = db.get_all_embeddings()
        in_persons = [p for p in all_persons if p['status'] == 1]
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls = track
            bbox = [x1, y1, x2, y2]
            
            # Draw bbox (default red)
            
            current_feature = reid.extract_features(frame_resized, bbox)
            
            best_match_id = None
            best_match_name = "Unknown"
            max_sim = 0.0
            entry_time = 0
            
            if current_feature is not None:
                # Check against ALL persons to ensure we identify them regardless of status
                # (e.g. if they missed entry scan or are testing)
                candidates = all_persons
                
                for person in candidates:
                    # Compare against ALL embeddings for this person
                    for emb in person['embeddings']:
                        sim = reid.compute_similarity(current_feature, emb)
                        if sim > max_sim:
                            max_sim = sim
                            best_match_id = person['id']
                            best_match_name = person['name']
                            entry_time = person.get('entry_time', 0)

            # Visualization Logic
            if max_sim > MATCH_THRESHOLD:
                # Valid Match
                color = (0, 255, 0) # Green
                label = f"{best_match_name} ({max_sim:.2f})"
                
                # Calculate Duration
                current_time = time.time()
                duration = current_time - entry_time if entry_time else 0
                
                # Log if not recently logged
                if best_match_id not in recent_exits or (current_time - recent_exits[best_match_id] > 10):
                    print(f"EXIT DETECTED: {best_match_name} (ID: {best_match_id})")
                    
                    # Format Duration
                    m, s = divmod(duration, 60)
                    h, m = divmod(m, 60)
                    duration_str = "{:d}:{:02d}:{:02d}".format(int(h), int(m), int(s))
                    
                    print(f"Duration: {duration_str}")
                    
                    # Update DB Status to OUT (0)
                    db.update_status(best_match_id, 0)
                    
                    # Log to CSV
                    with open("productivity_log.csv", "a") as f:
                        # Header: ID, Name, ExitTime, DurationSeconds, DurationFormatted
                        # If file empty, write header first? (Simplified for now)
                        exit_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                        f.write(f"{best_match_id},{best_match_name},{exit_time_str},{duration:.2f},{duration_str}\n")
                    
                    recent_exits[best_match_id] = current_time
                
                # Show Duration on screen
                # Check if we just logged it or it's in recent_exits
                if best_match_id in recent_exits and (current_time - recent_exits[best_match_id] < 5):
                     cv2.putText(frame_resized, f"EXIT: {duration:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Unknown / Stranger
                color = (0, 0, 255) # Red
                label = f"Stranger ({max_sim:.2f})"

            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame_resized, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Exit Camera", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    main()
