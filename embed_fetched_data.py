import sys
sys.path.append('../embedding_indexer')

import os
import sqlite3
from tqdm import tqdm
from src.fetcher import Fetcher
from sqlitedict import SqliteDict
from src.facenet import get_embedding_model
from src.general_utils import crop, preprocess
from config.fetcher_config import FETCHER_CONFIG
from config.common_config import COMMON_CONFIG
from emb_utils import EmbeddingIndex

FORCE_UPDATE = False
VECTOR_DB_CHECKPOINT = os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FacenetInception_v2")

if __name__ == '__main__':
    # utils
    fetcher = Fetcher(FETCHER_CONFIG)
    embedding_model = get_embedding_model(COMMON_CONFIG["embedding_model"])
    vector_db       = EmbeddingIndex(
                        **COMMON_CONFIG['ann_config']
                    )

    # load db
    media_items = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='media_items', autocommit=False)
    face_locs   = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='face_locations', autocommit=False)
    face_emb    = SqliteDict(os.path.join(COMMON_CONFIG["root"], "data/photos_db_v20231217/FaceClusterSQLDB.db"), tablename='face_embeddings', autocommit=False)
    
    if not FORCE_UPDATE and os.path.isfile(VECTOR_DB_CHECKPOINT):
        vector_db.load_index(VECTOR_DB_CHECKPOINT)

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

    for face_loc in tqdm(list(face_locs.keys())):
        if not FORCE_UPDATE and face_loc in face_emb:
            # found existing embedding for this (photo, location)
            continue

        # check if model is confident this is a person's face
        conf     = face_locs[face_loc]['score']
        if conf < COMMON_CONFIG['face_confidence_threshold']:
            continue
        
        # get image
        mediaId = face_locs[face_loc]['media_id']
        img = fetcher.fetchPhoto(mediaId=mediaId)

        # preprocess image
        box = face_locs[face_loc]['loc']
        face_cropped = preprocess(crop(img, box), rows=COMMON_CONFIG['face_shape'][0], cols=COMMON_CONFIG['face_shape'][1])

        # find emb
        emb = embedding_model.predict(face_cropped[None,...], verbose=0)

        # save embedding and assume unknown person
        cursor.execute('''
            INSERT INTO all_persons (id, mediaId, personId)
            VALUES (?, ?, ?)
        ''', (face_loc, mediaId, 'Unknown'))
        face_emb[face_loc] = {
            'embedding_model': COMMON_CONFIG["embedding_model"],
            "embedding": emb,
        }
        vector_db.add_embeddings(
            embeddings=emb, 
            metadata={
                'id': id,
            }
        )
    
        # checkpoint
        vector_db.save_index(VECTOR_DB_CHECKPOINT)
        connection.commit()
        face_emb.commit()

    media_items.close()
    face_locs.close()
    face_emb.close()
    connection.commit()
    connection.close()