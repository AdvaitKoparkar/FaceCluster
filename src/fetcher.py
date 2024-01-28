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

    def fetchMediaItems(self, request : dict) -> list :
        '''
            returns list of mediaItems with Google Photos API
        '''
        fetchedItems = []
        logger.debug('fetching request')

        for clientID in self.config.get("clientIDs"):
            logger.debug(f'connecting to {clientID} photos')
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

    def fetchAlbums(self, ) -> list :
        '''
            return albums & sharedAlbums from Google Photos API
        '''
        fetchedAlbums = []
        logger.debug('fetching request')

        for clientID in self.config.get("clientIDs"):
            logger.debug(f'connecting to {clientID} photos')
            service = photosCreateService(clientID)
            nextPageToken = ''
            while True:
                response = service.albums().list(pageToken=nextPageToken).execute()
                albums = response.get('albums')
                if albums is not None:
                    fetchedAlbums += albums
        
                nextPageToken = response.get("nextPageToken")
                if not nextPageToken:
                    break
            
            nextPageToken = ''
            while True:
                response = service.sharedAlbums().list(pageToken=nextPageToken).execute()
                albums = response.get('sharedAlbums')
                if albums is not None:
                    fetchedAlbums += albums
        
                nextPageToken = response.get("nextPageToken")
                if not nextPageToken:
                    break
        
        logger.debug(f'fetched {len(fetchedAlbums)} albums')
        return fetchedAlbums

    def fetchPhoto(self, mediaId : str ) -> np.ndarray :
        '''
            fetch photo (pixels) using mediaItem
        '''
        logger.debug(f'searching for requested mediaId')

        # search which id has the mediaItem
        mediaItem = None
        for clientID in self.config.get("clientIDs"):
            logger.debug(f'connecting to {clientID} photos')
            service = photosCreateService(clientID)
            mediaItem = service.mediaItems().get(mediaItemId=mediaId).execute()
            if mediaItem.get('id') == mediaId:
                logger.debug(f'found media item in {clientID}')
                break
            else:
                mediaItem = None

        if mediaItem is None:
            raise Exception(f'requested mediaId not found')

        baseUrl = mediaItem['baseUrl']
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
        return img

    # helpers
    def _updateMediaItem(self, mediaIds):
        updatedItems = []
        for clientID in self.config.get("clientIDs"):
            service = photosCreateService(clientID)
            response = service.mediaItems().batchGet(mediaIds)
            mediaItems = response.get("mediaItems")
            updatedItems += mediaItems
        return updatedItems
