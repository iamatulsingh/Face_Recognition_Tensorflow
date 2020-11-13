import os
from face_recognition import FaceRecognition


if __name__ == '__main__':
    model_path = "model"
    image_path = 'test.jpg'
    face_recognition = FaceRecognition()
    face_recognition.training()
    face_recognition.save_model(model_name)
    model = FaceRecognition.load_saved_model(model_path)
    k = FaceRecognition.model_prediction(image_path, model_path)
    print(f"detected class is {k}")
