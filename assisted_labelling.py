import sys
sys.path.append('../embedding_indexer')

import yaml
import argparse
from src.general_utils import AssistedLabeler
from src.fetcher import Fetcher
from config.common_config import COMMON_CONFIG
from config.fetcher_config import FETCHER_CONFIG
from emb_utils import EmbeddingIndex

_SUPPORTED_ACTIONS = {
    'check_status',
    'label_unknown',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI labelling for FaceCluster")
    parser.add_argument("action", action='store', 
                        help=f'suported actions {_SUPPORTED_ACTIONS}')
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

    # load fetcher
    fetcher = Fetcher(FETCHER_CONFIG)

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
    elif args.action == 'label_unknown':
        unknowns = labeller.get_unknown(n=4)
        for idx in range(len(unknowns)):
            labeller.view_face(unknowns[idx], fetcher)
            personId = input("Person Id: ")
            unknowns[idx]['personId'] = personId
        status = labeller.set_identity(unknowns)
        print(status)