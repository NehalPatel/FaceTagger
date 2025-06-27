from models.face_recognition_model import FaceRecModel
from models.insightface_model import InsightFaceModel
from models.deepface_model import DeepFaceModel
from deepface import DeepFace
import cv2
import os
import pickle
import numpy as np

def main(model_name="insightface"):
    if model_name == "face_recognition":
        model = FaceRecModel()
        enc_file = "encodings/face_recognition.pkl"
    elif model_name == "deepface":
        model = DeepFaceModel()
        enc_file = "encodings/deepface_arcface.pkl"
    else:
        model = InsightFaceModel()
        enc_file = "encodings/insightface.pkl"
    model.encode_known_faces("known_faces", enc_file)
    model.recognize_faces("test_images", enc_file, "output", model_name)

if __name__ == "__main__":
    main("insightface")  # or "face_recognition"
    main("face_recognition")  # or "face_recognition"
    main("deepface")
