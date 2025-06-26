# models/insightface_model.py
import os
import cv2
import numpy as np
import pickle
import insightface
from insightface.app import FaceAnalysis
from models.base import FaceRecognizer
from app.utils import list_images
from datetime import datetime

class InsightFaceModel(FaceRecognizer):
    def encode_known_faces(self, input_dir, output_path):
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        # app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # print("Execution Providers:", app.ctx.providers)
        app.prepare(ctx_id=0)

        known_encodings = []
        known_names = []

        for file in list_images(input_dir):
            path = os.path.join(input_dir, file)
            name = os.path.splitext(file)[0]
            img = cv2.imread(path)
            faces = app.get(img)

            if faces:
                known_encodings.append(faces[0].embedding)
                known_names.append(name)
                print(f"[INFO] Encoded: {name}")
            else:
                print(f"[WARN] No face found in {file}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((known_names, known_encodings), f)

    def recognize_faces(self, test_dir, encodings_path, output_dir):
        # app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0)

        with open(encodings_path, "rb") as f:
            known_names, known_encodings = pickle.load(f)

        os.makedirs(output_dir, exist_ok=True)

        for file in list_images(test_dir):
            path = os.path.join(test_dir, file)
            image = cv2.imread(path)
            faces = app.get(image)

            for face in faces:
                distances = [np.linalg.norm(face.embedding - enc) for enc in known_encodings]
                best_idx = np.argmin(distances)
                name = known_names[best_idx] if distances[best_idx] < 0.7 else "Unknown"

                box = face.bbox.astype(int)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(image, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename, ext = os.path.splitext(file)
            output_filename = f"result_{filename}_{timestamp}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image)
            print(f"[✅] Saved: {output_filename}")
