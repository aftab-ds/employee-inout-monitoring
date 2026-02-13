
import os
import cv2
import argparse
from src.database import Database
from src.reid import ReIdentifier
from src.detector import PersonDetector

def main():
    parser = argparse.ArgumentParser(description="Register persons from images in a directory.")
    parser.add_argument("--dir", type=str, default="registration_images", help="Directory containing person subfolders")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.")
        return

    db = Database()
    reid = ReIdentifier()
    # We use detector to make sure we crop the person correctly if needed, 
    # but ReIdentifier.extract_features already takes a bbox.
    # For registration images, we might assume the face is prominent, 
    # but let's use the detector for robustness if image is large.
    detector = PersonDetector()

    for person_name in os.listdir(args.dir):
        person_dir = os.path.join(args.dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"\nProcessing person: {person_name}")
        
        # Add person to DB (or get ID if exists)
        person_id = db.add_person(person_name)
        
        count = 0
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"  Skipping {img_name} (could not read)")
                continue

            # Detect person to get bbox
            tracks = detector.track(frame)
            
            if not tracks:
                # If detector fails (maybe it's just a face crop), try full image as bbox
                h, w, _ = frame.shape
                bbox = [0, 0, w, h]
            else:
                # Use the largest person detected
                tracks.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
                bbox = tracks[0][:4]

            embedding = reid.extract_features(frame, bbox)
            
            if embedding is not None:
                db.add_embedding(person_id, embedding)
                print(f"  Added embedding from {img_name}")
                count += 1
            else:
                print(f"  Failed to extract embedding from {img_name}")

        print(f"Finished {person_name}. Total embeddings added: {count}")

    db.close()

if __name__ == "__main__":
    main()
