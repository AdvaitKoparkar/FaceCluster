import logging
from tqdm import tqdm
from sqlitedict import SqliteDict

from src.fetcher import Fetcher
from src.face_extractor import FaceExtractor
from config.fetcher_config import FETCHER_CONFIG

if __name__ == '__main__':
    logger = logging.getLogger('src')
    logger.setLevel('DEBUG')

    # init face extractor
    face_extractor = FaceExtractor()

    # load db
    media_items = SqliteDict("data/photos_db_v20231217/FaceClusterSQLDB.db", tablename='media_items')
    fetcher = Fetcher(FETCHER_CONFIG)

    # create new db with face locations
    face_locs = SqliteDict("data/photos_db_v20231217/FaceClusterSQLDB.db", tablename='face_locations')
    for mediaId in tqdm(list(media_items.keys())):
        img = fetcher.fetchPhoto(mediaId)
        faces = face_extractor.get_faces(img)
        for face in faces:
            id = mediaId + '_' + face['id']
            face['media_id'] = mediaId
            face_locs[id] = face
        # checkpoint
        face_locs.commit()
    
    face_locs.close()
    media_items.close()