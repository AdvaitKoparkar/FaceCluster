from src.fetcher import Fetcher
from config.fetcher_config import FETCHER_CONFIG
import matplotlib.pyplot as plt

ff = Fetcher(FETCHER_CONFIG)
request = {
  "filters": {
    "dateFilter": {
      "ranges": [
        {
          "startDate": {
            "day": 1,
            "month": 1,
            "year": 2023
          },
          "endDate": {
            "day": 1,
            "month": 3,
            "year": 2023
          }
        }
      ]
    }
  }
}
mediaItems = ff.fetchMetaData(request=request)
for mediaItem in mediaItems:
    img = ff.fetchPhoto(mediaItem)
    plt.imshow(img)
    plt.show()