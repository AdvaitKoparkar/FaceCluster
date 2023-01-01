from config.mtcnn_config import MTCNN_CONFIG
from model.mtcnn_keras.mtcnn import MTCNN

m = MTCNN(MTCNN_CONFIG)
print(m.PNet.summary())
