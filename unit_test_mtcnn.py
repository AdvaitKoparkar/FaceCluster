import os
import cv2
import matplotlib.pyplot as plt

from config.mtcnn_config import MTCNN_CONFIG
from src.face_extractor import FaceExtractor

img = plt.imread(os.path.join(MTCNN_CONFIG["unit_test_data_path"], "test_face_001.jpeg"))

fe = FaceExtractor()
face_locs, scores = fe.get_faces(img)

for face_loc in face_locs:
    print(face_loc)
    img = cv2.rectangle(img, (face_loc[0], face_loc[1]), (face_loc[0]+face_loc[2], face_loc[1] + face_loc[3]), color=[0,1,0], thickness=2)

plt.imshow(img)
plt.show()