import mtcnn
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FaceExtractor(object):
    __VERSION__ = '20231217'
    def __init__(self, method = 'mtcnn'):
        if method == 'mtcnn':
            logger.debug('using pre-trained MTCNN face extrctor')
            self.detector = mtcnn.MTCNN()
    
    def get_faces(self, img : np.ndarray ) -> list :
        face_locs = self.detector.detect_faces(img)
        faces = []
        logger.debug(f'found {len(face_locs)}')
        for face_idx, face in enumerate(face_locs):
            logging.debug(f'face#{face_idx:03d} loc: {face["box"]}, score:{face["confidence"]:.02f}')
            faces.append({
                'version':  self.__VERSION__,
                'id':       f'{face_idx:03d}',
                'loc':      face['box'],
                'score':    face['confidence']
            })
        return faces