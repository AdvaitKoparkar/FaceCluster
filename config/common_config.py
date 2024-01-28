import os

PROJECTS_FOLDER = r"/Users/advaitkoparkar/Documents/projects"
COMMON_CONFIG = \
{
    "root": os.path.join(f"{PROJECTS_FOLDER}", "FaceCluster"),
    "face_shape": (160,160,3),
    "face_confidence_threshold": 0.98,
    "embedding_model": {
        "name": "FacenetInception_v2",
        "dimension": 128,
        "wts": os.path.join(PROJECTS_FOLDER, "FaceCluster", "src", "facenet_inception.h5"),
    },
}
