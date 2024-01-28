import json
import logging
from tqdm import tqdm
from sqlitedict import SqliteDict

from src.fetcher import Fetcher
from config.fetcher_config import FETCHER_CONFIG

_ALBUMS = [
    'Advait - Nidhi Getting Together Party'
]

if __name__ == '__main__':
    logger = logging.getLogger('src')
    logger
    logger.setLevel('DEBUG')

    # load db
    media_items = SqliteDict("data/photos_db_v20231217/FaceClusterSQLDB.db", tablename='media_items', autocommit=False)

    # update db with requested photos
    fetcher = Fetcher(FETCHER_CONFIG)
    albums = fetcher.fetchAlbums()
    for album_idx, album in enumerate(albums):
        album_id = album['id']
        if 'title' not in album.keys() or album['title'] not in _ALBUMS:
            continue
        
        # fetch all photos from album
        request = {'albumId': album_id}
        logger.debug(f'fetching album {album["title"]}')
        mediaItems = fetcher.fetchMediaItems(request)
        for mediaItem in tqdm(mediaItems):
            id = mediaItem['id']
            metadata = mediaItem['mediaMetadata']
            media_items[id] = metadata

        # save database
        media_items.commit()

    media_items.close()