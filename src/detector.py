
import cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """
        Initialize YOLOv8 detector.
        Args:
            model_path (str): Path to YOLOv8 model weights.
            conf_threshold (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Class ID for 'person' in COCO dataset is 0
        self.target_class_id = 0 

    def track(self, frame):
        """
        Track persons in a frame using ByteTrack.
        Args:
            frame (numpy.ndarray): Input image frame.
        Returns:
            list: List of tracks [x1, y1, x2, y2, track_id, score, class_id]
        """
        # specific tracker configuration can be passed if needed, defaulting to bytetrack
        results = self.model.track(frame, persist=True, verbose=False, conf=self.conf_threshold, classes=[self.target_class_id], tracker="bytetrack.yaml")
        
        tracks = []
        for result in results:
            boxes = result.boxes
            if boxes.id is not None:
                track_ids = boxes.id.cpu().numpy()
                xyxys = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                
                for xyxy, track_id, conf, cls in zip(xyxys, track_ids, confs, clss):
                    tracks.append([*xyxy, int(track_id), conf, cls])
            else:
                 # Fallback if no tracks specific but detections exist (rare with track mode)
                 pass
        
        return tracks
