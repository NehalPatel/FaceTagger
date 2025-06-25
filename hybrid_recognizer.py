import os
import cv2
import numpy as np
import pickle
from app.utils import list_images
import face_recognition
from insightface.app import FaceAnalysis
from models.face_recognition_model import FaceRecModel

def run_hybrid():
    enc_file = "encodings/face_recognition.pkl"
    # Ensure encodings exist
    model = FaceRecModel()
    model.encode_known_faces("known_faces", enc_file)
    with open(enc_file, "rb") as f:
        known_names, known_encodings = pickle.load(f)
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    for file in list_images("test_images"):
        path = os.path.join("test_images", file)
        image_bgr = cv2.imread(path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = app.get(image_bgr)
        print(f"[HYBRID] Detected {len(faces)} face(s) in {file}")
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            # Convert to (top, right, bottom, left) for face_recognition
            top, right, bottom, left = y1, x2, y2, x1
            encodings = face_recognition.face_encodings(image_rgb, [(top, right, bottom, left)])
            name = "Unknown"
            if encodings:
                face_encoding = encodings[0]
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                if matches:
                    best_match_index = np.argmin(distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, f"result_hybrid_{file}"), image_bgr)
        print(f"[HYBRID] Saved: {file}")

if __name__ == "__main__":
    run_hybrid()