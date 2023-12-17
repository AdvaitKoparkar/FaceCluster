import json
import logging
import argparse
from tqdm import tqdm
from sqlitedict import SqliteDict

from src.fetcher import Fetcher
from config.fetcher_config import FETCHER_CONFIG

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='PhotosFetcher', description='updates photos DB')
  parser.add_argument('--db', action='store', dest='db', 
                      help='path to SQL database')
  parser.add_argument('--request', action='store', dest='request', 
                      help='path to request json')
  args = parser.parse_args()

  logger = logging.getLogger('src')
  logger.setLevel('DEBUG')

  # read request from json
  with open(args.request, 'r') as fh:
    request = json.load(fh)
  # load db
  db = SqliteDict(args.db, tablename='media_items', autocommit=False)

  # update db with requested photos
  fetcher = Fetcher(FETCHER_CONFIG)
  mediaItems = fetcher.fetchMetaData(request)
  for mediaItem in tqdm(mediaItems):
    id = mediaItem['id']
    metadata = mediaItem['mediaMetadata']
    db[id] = metadata
  db.commit()
  db.close()
