import json
import logging
from tqdm import tqdm
from sqlitedict import SqliteDict

from src.fetcher import Fetcher
from config.fetcher_config import FETCHER_CONFIG

if __name__ == '__main__':
    logger = logging.getLogger('src')
    logger.setLevel('DEBUG')

    # read request from json
    with open("data/photos_db_v20231217/photos_request.json", 'r') as fh:
        request = json.load(fh)

    # load db
    media_items = SqliteDict("data/photos_db_v20231217/FaceClusterSQLDB.db", tablename='media_items', autocommit=False)

    # update db with requested photos
    fetcher = Fetcher(FETCHER_CONFIG)
    albums = fetcher.fetchAlbums()
    for album in tqdm(albums):
        album_id = album['id']
        request['albumId'] = album_id
        logger.debug(f'fetching album {album["title"]}')
        mediaItems = fetcher.fetchMediaItems(request)
        for mediaItem in tqdm(mediaItems):
            id = mediaItem['id']
            metadata = mediaItem['mediaMetadata']
            media_items[id] = metadata

    media_items.commit()
    media_items.close()