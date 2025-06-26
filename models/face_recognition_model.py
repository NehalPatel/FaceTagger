# models/face_recognition_model.py
import os
import cv2
import face_recognition
import pickle
from datetime import datetime
from models.base import FaceRecognizer
from app.utils import list_images

class FaceRecModel(FaceRecognizer):
    def encode_known_faces(self, input_dir, output_path):
        known_encodings = []
        known_names = []

        for file in list_images(input_dir):
            path = os.path.join(input_dir, file)
            name = os.path.splitext(file)[0]
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[INFO] Encoded: {name}")
            else:
                print(f"[WARN] No face found in {file}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((known_names, known_encodings), f)

    def recognize_faces(self, test_dir, encodings_path, output_dir):
        with open(encodings_path, "rb") as f:
            known_names, known_encodings = pickle.load(f)

        os.makedirs(output_dir, exist_ok=True)

        for file in list_images(test_dir):
            path = os.path.join(test_dir, file)
            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, locations)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            for (box, face_encoding) in zip(locations, encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                distances = face_recognition.face_distance(known_encodings, face_encoding)

                if matches:
                    best_match_index = distances.argmin()
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                top, right, bottom, left = box
                cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image_bgr, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename, ext = os.path.splitext(file)
            output_filename = f"result_{filename}_{timestamp}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image_bgr)
            print(f"[✅] Saved: {output_filename}")