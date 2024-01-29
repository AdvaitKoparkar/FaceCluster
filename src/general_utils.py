import cv2
import sqlite3
import numpy as np
from sqlitedict import SqliteDict

def crop(image : np.ndarray, crop : np.ndarray ) -> np.ndarray :
    return image[crop[1]:crop[1]+crop[3], crop[0]:crop[0] + crop[2]]

def preprocess(image : np.ndarray, **cfg) -> np.ndarray :
    result_image = image.copy()
    result_image = result_image.astype(np.float32)

    rows = cfg.get('rows', result_image.shape[0])
    cols = cfg.get('cols', result_image.shape[1])
    result_image = cv2.resize(result_image, (cols, rows), cv2.INTER_LINEAR)
    mean = result_image.mean()
    std  = result_image.std()
    result_image = (result_image - mean) / std
    
    return result_image

def draw_rectangle(image : np.ndarray , box : np.ndarray, text : str , **cfg) -> np.ndarray :
    thickness    = cfg.get('thickness', 1)
    color        = cfg.get('color', (0,255,0))
    font         = cfg.get('font', cv2.FONT_HERSHEY_SIMPLEX)
    font_scale   = cfg.get('font', 0.8)
    result_image = image.copy()

    cv2.rectangle(
        result_image, 
        (box[0], box[1]), 
        (box[0] + box[2], box[1] + box[3]),
        color=color,
        thickness=thickness,
    )

    cv2.putText(
        result_image, 
        text, 
        (box[0], box[1]), 
        font, font_scale, color, thickness)

    return result_image

# TODO: 
# https://qsl.robinbay.com/
# https://stackoverflow.com/questions/54921711/interactive-labeling-of-images-in-jupyter-notebook
class AssistedLabeler(object):
    def __init__(self, cfg : dict ):
        self.face_emb       = SqliteDict(**cfg['face_emb'])
        self.face_locs      = SqliteDict(**cfg['face_locs'])
        self.media_items    = SqliteDict(**cfg['media_items'])
        self.person_db      = cfg['person_db']
        self.vector_db      = cfg['vector_db']

    def check_status(self, ) -> dict :
        person_status = self._get_person_status()
        return person_status

    ## helpers
    def _get_person_status(self, ):
        status = {}
        connection = sqlite3.connect(self.person_db)
        cursor = connection.cursor()

        # TODO: make this part of COMMON_CONFIG
        person_table_name = "all_persons"
        person_id_col_name = "personId"

        # get number of people labelled Uknown
        cursor.execute(f'''
            SELECT COUNT(*) FROM {person_table_name} WHERE {person_id_col_name} = ?
        ''', ('Unknown',))
        status['unkown_count'] = cursor.fetchone()[0]

        # get total number of faces
        cursor.execute(f'''
            SELECT COUNT(*) FROM {person_table_name}
        ''')
        status['total_face_instances'] = cursor.fetchone()[0]

        # get distinct
        cursor.execute(f'''
            SELECT COUNT(DISTINCT {person_id_col_name}) FROM {person_table_name}
        ''')
        status['total_persons'] = cursor.fetchone()[0]

        connection.close()
        return status