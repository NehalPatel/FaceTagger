import os
import pickle
import face_recognition
import cv2
from .utils import list_images

def recognize_faces(test_dir='test_images', encodings_file='encodings/known_faces.pkl'):
    with open(encodings_file, "rb") as f:
        known_names, known_encodings = pickle.load(f)

    for file in list_images(test_dir):
        path = os.path.join(test_dir, file)
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        print(f"[🔍] Found {len(encodings)} face(s) in {file}")

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), face_encoding in zip(locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        output_path = os.path.join("output", f"result_{file}")
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, image_bgr)
        print(f"[💾] Saved result: {output_path}")
