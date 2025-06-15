import os
import pickle
import face_recognition
from .utils import list_images

def encode_known_faces(known_dir='known_faces', output_dir='encodings'):
    known_encodings = []
    known_names = []

    for file in list_images(known_dir):
        path = os.path.join(known_dir, file)
        name = os.path.splitext(file)[0]

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"[INFO] Encoded: {name}")
        else:
            print(f"[WARN] No face found in {file}, skipping.")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "known_faces.pkl"), "wb") as f:
        pickle.dump((known_names, known_encodings), f)
    print(f"[✅] Saved {len(known_names)} encodings.")
