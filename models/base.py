from abc import ABC, abstractmethod

class FaceRecognizer(ABC):
    @abstractmethod
    def encode_known_faces(self, input_dir, output_path):
        pass

    @abstractmethod
    def recognize_faces(self, test_dir, encodings_path, output_dir):
        pass
