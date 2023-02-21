import os
from functools import partial
from skimage.transform import resize
from .common_config import COMMON_CONFIG

MTCNN_CONFIG = { 
    "unit_test_data_path": os.path.join(COMMON_CONFIG["root"], "data", "unit_test_data", "mtcnn_unit_test_data"),
    "PNet_weights": os.path.join(COMMON_CONFIG['root'], 'model', 'mtcnn_keras', 'weights', '12net.h5'),
    "PNet_prediction_NMS_config": {
        "threshold": 0.3
    },
    "RNet_weights": os.path.join(COMMON_CONFIG['root'], 'model', 'mtcnn_keras', 'weights', '24net.h5'),
    "ONet_weights": os.path.join(COMMON_CONFIG['root'], 'model', 'mtcnn_keras', 'weights', '48net.h5'),
    "image_scaling_config": { 
        "max_dim": 700,
        "min_dim": 128,
        "recursive_scaling_factor": 0.90,
    },
    "image_resizing_fn": partial(resize, order=1, mode="reflect")
}
