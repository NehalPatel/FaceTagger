import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from models.base import FaceRecognizer

class DeepFaceModel(FaceRecognizer):
    def encode_known_faces(self, input_dir, output_path):
        known_encodings = []
        known_names = []
        for file in os.listdir(input_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(input_dir, file)
                name = os.path.splitext(file)[0]
                # Get embedding using DeepFace
                embedding = DeepFace.represent(img_path=path, model_name='ArcFace', enforce_detection=False)[0]["embedding"]
                known_encodings.append(np.array(embedding))
                known_names.append(name)
                print(f"[INFO] Encoded: {name}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((known_names, known_encodings), f)

    def recognize_faces(self, test_dir, encodings_path, output_dir, model_name="DeepFaceModel"):
        with open(encodings_path, "rb") as f:
            known_names, known_encodings = pickle.load(f)
        os.makedirs(output_dir, exist_ok=True)
        threshold = 10  # ArcFace embeddings are not normalized, so use a large threshold
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(test_dir, file)
                image = cv2.imread(path)
                try:
                    embedding = DeepFace.represent(img_path=path, model_name='ArcFace', enforce_detection=False)[0]["embedding"]
                    embedding = np.array(embedding)
                    distances = [np.linalg.norm(embedding - enc) for enc in known_encodings]
                    best_idx = np.argmin(distances)
                    name = known_names[best_idx] if distances[best_idx] < threshold else "Unknown"
                except Exception as e:
                    print(f"[WARN] No face found in {file} or error: {e}")
                    name = "Unknown"
                # Draw label
                cv2.putText(image, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                output_path = os.path.join(output_dir, f"result_{file}")
                cv2.imwrite(output_path, image)
                print(f"[✅] Saved: {output_path}") 