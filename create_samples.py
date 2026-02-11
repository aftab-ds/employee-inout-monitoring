
import cv2
import numpy as np

def create_dummy_video(filename, text="Person", duration=5):
    """
    Create a dummy video with a moving rectangle to simulate a person.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    
    for i in range(20 * duration):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Move rectangle
        x = int(100 + i * 2) % 500
        y = 240
        
        # Draw "Person"
        cv2.rectangle(frame, (x, y), (x+50, y+100), (0, 255, 0), -1)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        
    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_video("entry_cam.mp4", "Entry Person", 10)
    create_dummy_video("exit_cam.mp4", "Exit Person", 10)
