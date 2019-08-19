import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import cv2
from constant import Constant
from align import align_function
image_name= 'imgc.jpg'
base_dir= os.getcwd()+"/"
image_file = base_dir+"/img.jpg"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
top=0
right=0
bottom=0
left=0
name = ""
def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    if len(X_face_locations) == 0:
        return []
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    del draw
    pil_image.show()

def use_live(name,top,left,bottom,right):
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        cv2.imwrite(image_file, small_frame)
        #full_file_path = os.path.join(base_dir + "/face_image/test", image_file)
        full_file_path_test = align_function.get_crop_image(base_dir,112,image_file,image_name)
        full_file_path = os.path.join(base_dir + "/face_image/test", image_file)
        # print("Looking for faces in {}".format(image_file))


        predictions = predict(full_file_path, model_path=base_dir + "/trained_knn_model.clf")
        for name, (top, right, bottom, left) in predictions:
            pass
            # print("- Found {} at ({}, {})".format(name, left, top))
        # show_prediction_labels_on_image(os.path.join(base_dir + "face_image/test", image_file), predictions)
        #cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def use_all_image():
    for image_file in os.listdir(base_dir+"/face_image/test"):
        full_file_path = os.path.join(base_dir+"/face_image/test", image_file)
        print("Looking for faces in {}".format(image_file))
        predictions = predict(full_file_path, model_path=base_dir+"/trained_knn_model.clf")
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
        show_prediction_labels_on_image(os.path.join(base_dir+"face_image/test", image_file), predictions)
def use_image():
        image_file = base_dir+"/face_image/test/borhan.jpeg"
        full_file_path = os.path.join(base_dir+"/face_image/test", image_file)
        print("Looking for faces in {}".format(image_file))
        predictions = predict(full_file_path, model_path=base_dir+"/trained_knn_model.clf")
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
        show_prediction_labels_on_image(os.path.join(base_dir+"face_image/test", image_file), predictions)


def use_live_faster(name,top,left,bottom,right):
    video_capture = cv2.VideoCapture(0)
    knn_clf = None
    with open(base_dir + "/trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)
    distance_threshold = 0.6
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        X_img = face_recognition.load_image_file(base_dir+"/img.jpg")
        X_face_locations = face_recognition.face_locations(X_img)
        #X_face_locations = face_recognition.face_locations(rgb_small_frame)

        if len(X_face_locations) == 0:
            print ("None")

        #face_locations = face_recognition.face_locations(rgb_small_frame)
        #faces_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=X_face_locations)
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        print("Match ",are_matches)

        # full_file_path = os.path.join(base_dir + "/face_image/test", image_file)
        # # print("Looking for faces in {}".format(image_file))
        # predictions = predict(full_file_path, model_path=base_dir + "/trained_knn_model.clf")
        # for name, (top, right, bottom, left) in predictions:
        #     pass
        #     # print("- Found {} at ({}, {})".format(name, left, top))
        # # show_prediction_labels_on_image(os.path.join(base_dir + "face_image/test", image_file), predictions)
        # # cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        # cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        # cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        # cv2.imshow('Video', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

if __name__ == "__main__":
        #use_image()
        use_live(name,top,left,bottom,right)
        #use_live_faster(name,top,left,bottom,right)