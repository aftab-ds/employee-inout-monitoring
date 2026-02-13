
### Stage	Model	Input	Output	Role
1. Detection	YOLOv8n	Full Video Frame	Person Bounding Box	Finds the "human" in the scene and tracks them across frames.
2. Alignment	MTCNN	Crop of the Person	Cropped & Aligned Face	"Zooms in" on the head and aligns the eyes/nose to prepare for recognition.
3. Recognition	InceptionResnetV1	Aligned Face Image	512-dim Feature Vector	Converts the face into a unique mathematical signature (embedding).
   
### Why this "3-Model" approach is good:
Precision: By using YOLO first, you ensure you only look for faces where people actually are, which prevents false positives from background patterns.

Robustness: MTCNN allows the system to handle people even if they aren't looking directly at the camera (it can find the face within the body crop).

Modular: You could swap any of these (e.g., upgrade YOLOv11 or use a different FaceNet model) without rewriting the entire system.

In short: YOLO finds the person, MTCNN finds their face, and InceptionResnet remembers who they are.

### For Registering a Person
1. Prepare Images: Create a folder named registration_images. Inside it, create subfolders for each person (e.g., registration_images/Person1/).

2. Add Images: Place several photos of each person in their respective subfolder (different angles, lighting, etc.).

3. Run Registration:
   
python register_persons.py

4. Verify: Check the database list:

python manage_db.py --list

