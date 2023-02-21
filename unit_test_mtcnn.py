import os
import matplotlib.pyplot as plt

from config.mtcnn_config import MTCNN_CONFIG
from model.mtcnn_keras.mtcnn import MTCNN

test_img_fname = os.path.join(MTCNN_CONFIG['unit_test_data_path'], "test_face_000.jpeg")
img = plt.imread(test_img_fname)

# fig, ax = plt.subplots(1,1)
# ax.imshow(img)
# plt.show()

m = MTCNN(MTCNN_CONFIG)
m.detect_faces(img)
