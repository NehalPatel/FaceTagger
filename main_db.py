import os
import sqlite3
import cv2
import numpy as np
import face_recognition
from app.utils import list_images
from models.face_recognition_model import FaceRecModel

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL
                )''')
    conn.commit()
    return conn

def encode_known_faces_to_db(input_dir, db_path):
    conn = create_db(db_path)
    c = conn.cursor()
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for img_file in list_images(person_dir):
            img_path = os.path.join(person_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0].tobytes()
                c.execute("INSERT INTO encodings (name, encoding) VALUES (?, ?)", (person, encoding))
                print(f"[DB] Encoded and saved: {person} - {img_file}")
            else:
                print(f"[WARN] No face found in {img_file}")
    conn.commit()
    conn.close()

def main():
    lfw_dir = "dataset/LFW/lfw-deepfunneled/lfw-deepfunneled"
    db_path = "encodings/lfw_encodings.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    encode_known_faces_to_db(lfw_dir, db_path)

if __name__ == "__main__":
    main()