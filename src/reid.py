
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

class ReIdentifier:
    def __init__(self):
        """
        Initialize Face Recognition model.
        Uses MTCNN for face detection and InceptionResnetV1 for embedding.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MTCNN for face detection (keep_all=False means return best face)
        try:
            self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=self.device)
            # InceptionResnetV1 pretrained on VGGFace2
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("Face Recognition Models Loaded (MTCNN + InceptionResnetV1)")
        except Exception as e:
            print(f"Error loading Face Recognition models: {e}")
            self.mtcnn = None
            self.resnet = None

    def extract_features(self, frame, bbox):
        """
        Extract Face features from a person crop.
        Args:
            frame (numpy.ndarray): Full image frame.
            bbox (list): Person Bounding box [x1, y1, x2, y2].
        Returns:
            numpy.ndarray: 512-dim feature vector, or None if no face found.
        """
        if self.mtcnn is None or self.resnet is None:
            return None

        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return None

        person_crop = frame[y1:y2, x1:x2]
        
        try:
            # Convert to PIL for MTCNN
            img = Image.fromarray(person_crop[..., ::-1]) # BGR to RGB
            
            # Detect and crop face
            # mtcnn(img) returns tensor of shape (3, 160, 160)
            face_tensor = self.mtcnn(img)
            
            if face_tensor is not None:
                # Add batch dim
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                # Embedding
                with torch.no_grad():
                    embedding = self.resnet(face_tensor).squeeze().cpu().numpy()
                
                # Normalize (Facenet output is usually normalized, but let's be safe)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
                return embedding
            else:
                # No face detected in this person crop
                return None
        except Exception as e:
            # print(f"ReID Error: {e}") 
            return None

    @staticmethod
    def compute_similarity(feat1, feat2):
        """
        Compute Cosine Similarity between two feature vectors.
        """
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2)
