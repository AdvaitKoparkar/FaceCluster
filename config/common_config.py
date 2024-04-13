import os

PROJECTS_FOLDER = r"/Users/advaitkoparkar/Documents/projects"
_EMBEDDING_DIMENSION = 128
COMMON_CONFIG = \
{
    "root": os.path.join(f"{PROJECTS_FOLDER}", "FaceCluster"),
    "face_shape": (160,160,3),
    "face_confidence_threshold": 0.98,
    "embedding_model": {
        "name": "FacenetInception_v2",
        "dimension": _EMBEDDING_DIMENSION,
        "wts": os.path.join(PROJECTS_FOLDER, "FaceCluster", "src", "facenet_inception.keras"),
    },
    "ann_config": {
        "dimension": _EMBEDDING_DIMENSION,
        "index_params": {
            "index_type": "FlatIP",
        }
    },
    "default_labelling_config": os.path.join(f"{PROJECTS_FOLDER}", "FaceCluster", "config", "labelling_config.yml"),
}
