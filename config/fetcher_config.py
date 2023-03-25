import os
from .common_config import COMMON_CONFIG

FETCHER_CONFIG = \
{
    "token_path": os.path.join(COMMON_CONFIG["root"], "data", "photos_tokens"),
    "clientIDs": ['ak'],
    "mediaIdCacheSize": 1024,
    "mediaItemCacheSize": 1024,
    "mediaItemCacheTTL": 3600,
    "maxRetryAttempts": 5,
}
