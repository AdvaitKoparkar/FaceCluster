import sys
sys.path.append('../embedding_indexer')

import os
import numpy as np
import matplotlib.pyplot as plt
from sqlitedict import SqliteDict
from src.fetcher import Fetcher
from src.face_extractor import FaceExtractor
from src.facenet import get_embedding_model
from config.common_config import COMMON_CONFIG
from config.fetcher_config import FETCHER_CONFIG
from emb_utils import EmbeddingIndex

from src.general_utils import preprocess, draw_rectangle, crop

class ImageSearch(object):
    def __init__(self, ):
        self.fetcher = Fetcher(FETCHER_CONFIG)
        self.embedding_model = get_embedding_model(COMMON_CONFIG["embedding_model"])
        self.fe = FaceExtractor()
        self.vector_db = EmbeddingIndex(**COMMON_CONFIG['ann_config'])
        self.vector_db.load_index(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FacenetInception_v2"))
        self.media_items = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='media_items', autocommit=False)
        self.face_locs   = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='face_locations', autocommit=False)
        self.face_emb    = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='face_embeddings', autocommit=False)

    def search(self, img : np.ndarray, k = 3 ) -> list :
        faces = self.fe.get_faces(img)
        search_results = []
        for face in faces:
            box = face['loc']
            face_cropped = preprocess(crop(img, box), rows=COMMON_CONFIG['face_shape'][0], cols=COMMON_CONFIG['face_shape'][1])
            emb = self.embedding_model.predict(face_cropped[None,...], verbose=0)
            dist, _, metadata = self.vector_db.search(emb, k)
            imgs = [None] * len(metadata[0])
            for _k in range(len(metadata[0])):
                metadata[0][_k]['dist'] = dist[0][_k]
                metadata[0][_k]['face_loc_info'] = self.face_locs[metadata[0][_k]['id']]
                metadata[0][_k]['media_info'] = self.media_items[self.face_locs[metadata[0][_k]['id']]['media_id']]
                imgs[_k] = self.fetcher.fetchPhoto(self.face_locs[metadata[0][_k]['id']]['media_id'])

            search_results.append({
                'query': {
                    'face_extractor_output': face,
                    'embedding': emb,
                },
                'ann_result': {
                    'metadata': metadata,
                    'imgs': imgs,
                }
            })
        
        return search_results


if __name__ == '__main__':
    query = '/Users/advaitkoparkar/Documents/projects/FaceCluster/data/IMG20231231142101.jpg'
    img = plt.imread(query)
    k = 3
    search_results = ImageSearch().search(img, k)

    fig, ax = plt.subplots(len(search_results), k+1, squeeze=False)
    for qidx in range(len(search_results)):
        ax[qidx][0].imshow(
            draw_rectangle(img, search_results[qidx]['query']['face_extractor_output']['loc'], '')
        )
        for ridx in range(k):
            ax[qidx][ridx+1].imshow(
                draw_rectangle(search_results[qidx]['ann_result']['imgs'][ridx], 
                               search_results[qidx]['ann_result']['metadata'][0][ridx]['face_loc_info']['loc'], '')
            )
    plt.show()