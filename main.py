from models.face_recognition_model import FaceRecModel
from models.insightface_model import InsightFaceModel

def main(model_name="insightface"):
    if model_name == "face_recognition":
        model = FaceRecModel()
        enc_file = "encodings/face_recognition.pkl"
    else:
        model = InsightFaceModel()
        enc_file = "encodings/insightface.pkl"

    model.encode_known_faces("known_faces", enc_file)
    model.recognize_faces("test_images", enc_file, "output")

if __name__ == "__main__":
    main("insightface")  # or "face_recognition"
    # main("face_recognition")  # or "face_recognition"
