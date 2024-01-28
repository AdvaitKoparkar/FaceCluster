import sys
sys.path.append('../embedding_indexer')

import yaml
import argparse
from src.general_utils import AssistedLabeler
from config.common_config import COMMON_CONFIG
from emb_utils import EmbeddingIndex

_SUPPORTED_ACTIONS = {
    'check_status'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI labelling for FaceCluster")
    parser.add_argument("action", action='store', 
                        help='check labelling status')
    parser.add_argument("--config", default=COMMON_CONFIG['default_labelling_config'], action='store',
                        required=False, help="path to the target resource.")
    args = parser.parse_args()

    # load config
    with open(args.config, 'rb') as fh:
        labelling_config = yaml.safe_load(fh)

    # load vector db
    vector_db = EmbeddingIndex(
                    **COMMON_CONFIG['ann_config']
                )
    vector_db.load_index(labelling_config['DB_PATHS']['VECTOR_DB'])

    # init labeller
    labeller = AssistedLabeler({
        'face_emb': {
            'filename': labelling_config['DB_PATHS']['DB_PATH'],
            'tablename': 'face_embeddings', 
            'autocommit': False,
        },
        'face_locs': {
            'filename': labelling_config['DB_PATHS']['DB_PATH'],
            'tablename': 'face_locations', 
            'autocommit': False,
        },
        'media_items': {
            'filename': labelling_config['DB_PATHS']['DB_PATH'],
            'tablename': 'media_items', 
            'autocommit': False,
        },
        'person_db': labelling_config['DB_PATHS']['PERSON_DB'],
        'vector_db': vector_db,
    })

    # perform action
    if args.action == 'check_status':
        print(labeller.check_status())