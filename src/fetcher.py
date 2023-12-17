import logging
import requests
import numpy as np
from io import BytesIO
from imageio import imread
from src.photos_utils import photosCreateService

logger = logging.getLogger(__name__)

class Fetcher(object):
    '''
        utility to fetch photos & metadata from Google Photos API
    '''
    def __init__(self, config):
        self.config = config

    def fetchMetaData(self, request : dict) -> list :
        '''
            returns list of mediaItems from Google Photos API
        '''
        fetchedItems = []
        logging.debug('fetching request')

        for clientID in self.config.get("clientIDs"):
            logging.debug(f'connecting to {clientID} photos')
            service = photosCreateService(clientID)

            while True:
                response = service.mediaItems().search(body=request).execute()
                mediaItems = response.get("mediaItems")
                if mediaItems is not None:
                    fetchedItems += mediaItems
                
                nextPageToken = response.get("nextPageToken")
                if not nextPageToken:
                    break
                request['pageToken'] = nextPageToken

        logger.debug(f'fetched {len(fetchedItems)} mediaItems')
        return fetchedItems

    def fetchPhoto(self, mediaItem : dict ) -> np.ndarray :
        '''
            fetch photo (pixels) using mediaItem
        '''
        mediaId = mediaItem['id']
        baseUrl = mediaItem['baseUrl']
        logger.info(f'fetching {mediaId} from {baseUrl}')

        response = requests.get(baseUrl)
        retryAttempts = 0
        while response.ok is False and retryAttempts < self.config.get("maxRetryAttempts"):
            mediaItems = self._updateMediaItem([mediaId])
            if len(mediaItems) == 0:
                raise Exception(f'could not find media item {mediaId}')
            baseUrl = mediaItems[0]['baseUrl']
            response = requests.get(baseUrl)
            retryAttempts += 1

        if response.ok is False:
            raise Exception(f'could not find media item {mediaId}')

        img = imread(BytesIO(response.content))
        self.mediaIdCache[mediaId] = img
        self.mediaItemCache[baseUrl] = img
        return img

    # helpers
    def _updateMediaItem(self, mediaIds):
        updatedItems = []
        for clientID in self.config.get("clientIDs"):
            service = photosCreateService(clientID)
            response = service.mediaItems.batchGet(mediaIds)
            mediaItems = response.get("mediaItems")
            updatedItems += mediaItems
        return updatedItems
