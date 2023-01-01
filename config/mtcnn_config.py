import os
from .common_config import COMMON_CONFIG

MTCNN_CONFIG = { \
    "PNet_weights": os.path.join(COMMON_CONFIG['root'], 'model', 'mtcnn_keras', 'weights', '12net.h5'),
    "RNet_weights": os.path.join(COMMON_CONFIG['root'], 'model', 'mtcnn_keras', 'weights', '24net.h5'),
    "ONet_weights": os.path.join(COMMON_CONFIG['root'], 'model', 'mtcnn_keras', 'weights', '48net.h5'),
}
