import sys
sys.path.append('../embedding_indexer')

import os
import sqlite3
import numpy as np
import pickle as pkl
from tqdm import tqdm
import tensorflow as tf
from src.fetcher import Fetcher
from sqlitedict import SqliteDict
from sklearn.preprocessing import Normalizer
from src.general_utils import crop, preprocess
from config.fetcher_config import FETCHER_CONFIG
from config.common_config import COMMON_CONFIG
from emb_utils import EmbeddingIndex

VECTOR_DB            = os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FacenetInception_v2")

if __name__ == '__main__':
    # utils
    fetcher = Fetcher(FETCHER_CONFIG)
    embedding_model = tf.keras.models.load_model(COMMON_CONFIG['embedding_model']['wts'], safe_mode=False)
    vector_db       = EmbeddingIndex(
                        **COMMON_CONFIG['ann_config']
                    )

    # load db
    media_items = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='media_items', autocommit=False)
    face_locs   = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='face_locations', autocommit=False)
    face_emb    = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='face_embeddings', autocommit=False)

    # create a sql db to store: 
    # id: unique id for each predicted face      (primary key)
    # media_id: which photo does this belong to
    # person_id: name of the person
    connection   = sqlite3.connect(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/Persons.sqlite"))
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS all_persons (
            id TEXT PRIMARY KEY,
            mediaId TEXT,
            personId TEXT
        )
    ''')

    embs, metadata = [], []
    for face_loc in tqdm(list(face_locs.keys())):
        # check if model is confident this is a person's face
        conf     = face_locs[face_loc]['score']
        if conf < COMMON_CONFIG['face_confidence_threshold']:
            continue
        
        # get image
        mediaId = face_locs[face_loc]['media_id']
        img = fetcher.fetchPhoto(mediaId=mediaId)

        # preprocess image
        box = face_locs[face_loc]['loc']
        max_dim = max(box[2], box[3])
        box[2], box[3] = max_dim, max_dim
        face_cropped = preprocess(crop(img, box), rows=COMMON_CONFIG['face_shape'][0], cols=COMMON_CONFIG['face_shape'][1])

        # find emb
        emb = embedding_model.predict(face_cropped[None,...], verbose=0).reshape((1, -1))

        # save embedding and assume unknown person
        is_present = cursor.execute(
            '''
            SELECT COUNT(*) FROM all_persons WHERE id = ?
            ''', (face_loc,)).fetchone()
        if is_present[0] == 0:
            cursor.execute('''
                INSERT INTO all_persons (id, mediaId, personId)
                VALUES (?, ?, ?)
            ''', (face_loc, mediaId, 'Unknown'))
        
        # face_loc : embedding
        face_emb[face_loc] = {
            'embedding_model': COMMON_CONFIG["embedding_model"],
            "embedding": emb,
        }

        # emb : face_loc
        embs.append(emb)
        metadata.append({
                'id': face_loc,
        })

        # checkpoint
        connection.commit()
        face_emb.commit()

    embs = np.concatenate(embs, axis=0)
    embs = Normalizer('l2').transform(embs)
    vector_db.add_embeddings(
        embeddings=embs, 
        metadata=metadata,
    )
    vector_db.save_index(VECTOR_DB)
    connection.commit()
    face_emb.commit()

    media_items.close()
    face_locs.close()
    face_emb.close()
    connection.commit()
    connection.close()