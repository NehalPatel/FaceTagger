import os
import random
import numpy as np
import cv2
import face_recognition
import pickle
import mysql.connector
from app.utils import list_images
from models.face_recognition_model import FaceRecModel
from models.insightface_model import InsightFaceModel

# ---- MySQL Connection Details ----
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'face_tagger',
    'port': 3306  # Change if not default
}

# ---- Table Creation ----
TABLES = {
    'facerecognition': '''CREATE TABLE IF NOT EXISTS encodings_facerecognition (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        encoding BLOB NOT NULL
    )''',
    'insightface': '''CREATE TABLE IF NOT EXISTS encodings_insightface (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        encoding BLOB NOT NULL
    )''',
    'hybrid': '''CREATE TABLE IF NOT EXISTS encodings_hybrid (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        encoding BLOB NOT NULL
    )'''
}

# ---- Utility Functions ----
def get_mysql_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def create_table(model_name):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(TABLES[model_name])
    conn.commit()
    cursor.close()
    conn.close()

def insert_encoding(model_name, name, encoding):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    table = f'encodings_{model_name}'
    sql = f"INSERT INTO {table} (name, encoding) VALUES (%s, %s)"
    cursor.execute(sql, (name, encoding))
    conn.commit()
    cursor.close()
    conn.close()

# ---- Encoding Functions ----
def encode_faces_facerecognition(input_dir):
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
                insert_encoding('facerecognition', person, encoding)
                print(f"[MySQL] Encoded (face_recognition): {person} - {img_file}")
            else:
                print(f"[WARN] No face found in {img_file}")

def encode_faces_insightface(input_dir):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for img_file in list_images(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            faces = app.get(img)
            if faces:
                encoding = faces[0].embedding.tobytes()
                insert_encoding('insightface', person, encoding)
                print(f"[MySQL] Encoded (insightface): {person} - {img_file}")
            else:
                print(f"[WARN] No face found in {img_file}")

def encode_faces_hybrid(input_dir):
    # Hybrid: insightface for detection, face_recognition for encoding
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for img_file in list_images(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = app.get(img_bgr)
            if faces:
                box = faces[0].bbox.astype(int)
                x1, y1, x2, y2 = box
                top, right, bottom, left = y1, x2, y2, x1
                encodings = face_recognition.face_encodings(img_rgb, [(top, right, bottom, left)])
                if encodings:
                    encoding = encodings[0].tobytes()
                    insert_encoding('hybrid', person, encoding)
                    print(f"[MySQL] Encoded (hybrid): {person} - {img_file}")
                else:
                    print(f"[WARN] No face found in {img_file} (hybrid)")
            else:
                print(f"[WARN] No face detected in {img_file} (hybrid)")

# ---- Random Test Set Selection ----
def get_random_people(input_dir, n=10):
    people = [p for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))]
    return random.sample(people, n)

def main():
    lfw_dir = "dataset/LFW/lfw-deepfunneled/lfw-deepfunneled"
    # Create tables
    for model in ['facerecognition', 'insightface', 'hybrid']:
        create_table(model)
    # Encode and save to MySQL
    print("Encoding with face_recognition...")
    encode_faces_facerecognition(lfw_dir)
    print("Encoding with insightface...")
    encode_faces_insightface(lfw_dir)
    print("Encoding with hybrid...")
    encode_faces_hybrid(lfw_dir)
    # For testing, select 10 random people
    test_people = get_random_people(lfw_dir, 10)
    print(f"Randomly selected people for testing: {test_people}")
    # You can now use test_people for further testing scripts

if __name__ == "__main__":
    main()