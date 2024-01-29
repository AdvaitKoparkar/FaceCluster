from .src.face_extractor import FaceExtractor
from .src.facenet import get_embedding_model
from .src.general_utils import crop, preprocess
from .config.common_config import COMMON_CONFIG
from flask import Flask, request, jsonify

# delete later
import matplotlib.pyplot as plt
img = plt.imread("./data/tmp.jpeg")
facenet = get_embedding_model(COMMON_CONFIG['embedding_model'])
fe = FaceExtractor()

app = Flask(__name__)
@app.get("/embed")
def process_image():
    result = []
    face_locs = fe.get_faces(img)
    for face_loc in face_locs:
        face_crop = crop(img, face_loc['loc'])
        face_pp   = preprocess(face_crop, rows=COMMON_CONFIG['face_shape'][0], cols=COMMON_CONFIG['face_shape'][1])
        emb       = facenet.predict(face_pp[None,...], verbose=0)

        result.append({
            "face_loc": face_loc,
            "emb": {
                "name": COMMON_CONFIG["embedding_model"]["name"],
                "embedding": emb.flatten().tolist(),
            }
        })

    return jsonify(result)