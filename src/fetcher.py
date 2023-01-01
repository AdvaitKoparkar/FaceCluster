import requests
from io import BytesIO
from imageio import imread
from cachetools import LFUCache, TTLCache

from .photos_utils import photosCreateService

class Fetcher(object):
    def __init__(self, config):
        self.config = config
        self.mediaIdCache = LFUCache(maxsize=config['mediaIdCacheSize'])
        self.mediaItemCache = TTLCache(maxsize=config['mediaItemCacheSize'], ttl=config['mediaItemCacheTTL'])

    def fetchMetaData(self, request):
        fetchedItems = []
        for clientID in self.config.get("clientIDs"):
            service = photosCreateService(clientID)
            while True:
                response = service.mediaItems().search(body=request).execute()
                mediaItems = response.get("mediaItems")
                fetchedItems += mediaItems

                nextPageToken = response.get("nextPageToken")
                if not nextPageToken:
                    break

        return fetchedItems

    def updateMediaItem(self, mediaIds):
        updatedItems = []
        for clienID in self.config.get("clientIDs"):
            service = photosCreateService(clientID)
            response = service.mediaItems.batchGet(mediaIds)
            mediaItems = response.get("mediaItems")
            updatedItems += mediaItems
        return updatedItems

    def fetchPhoto(self, mediaItem):
        mediaId = mediaItem['id']
        baseUrl = mediaItem['baseUrl']
        if mediaId in self.mediaIdCache:
            return self.mediaIdCache[mediaId]
        if baseUrl in self.mediaItemCache:
            self.mediaIdCache[mediaId] = self.mediaItemCache[baseUrl]
            return self.mediaItemCache[baseUrl]

        response = requests.get(baseUrl)
        retryAttempts = 0
        while response.ok is False and retryAttempts < self.config.get("maxRetryAttempts"):
            mediaItems = self.updateMediaItem([mediaId])
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
