import os
from face_recognition import FaceRecognition


if __name__ == '__main__':
    model_name = "face_recognition.h5"
    image_path = 'test.jpg'
    face_recognition = FaceRecognition()
    # face_recognition.training()
    # face_recognition.save_model(model_name)
    # model = FaceRecognition.load_saved_model(os.path.join("model", model_name))
    k = FaceRecognition.model_prediction(image_path, os.path.join("model", model_name),
                                         os.path.join("model", "face_recognition_class_names.npy"))
    print(f"detected class is {k}")
