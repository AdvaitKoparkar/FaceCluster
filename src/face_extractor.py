import mtcnn
import logging
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)

class FaceRecognitionWrapper(object):
    def __init__(self, **kwargs):
        self.cfg = {}
        self.cfg['model'] = kwargs.get('model', 'hog')
    
    def _preprocess(self, img_in : np.ndarray ) -> np.ndarray :
        img     = img_in.copy()
        img     = img.astype(np.float32)
        img    -= img.min(axis=(0,1), keepdims=True)
        img    /= img.max(axis=(0,1), keepdims=True)
        img    *= 255
        img    = img.astype(np.uint8)
        return img

    def detect_faces(self, img : np.ndarray) -> list :
        img = self._preprocess(img)
        face_locs_fl = face_recognition.face_locations(img, **self.cfg)
        
        face_locs = []
        for face_loc_fl in face_locs_fl:
            [top, right, bottom, left] = face_loc_fl
            face_locs.append({
                'box': [left, top, right-left, bottom-top],
                'confidence': 1.0,
            })
        return face_locs

class FaceExtractor(object):
    __VERSION__ = '20231217'
    def __init__(self, method = 'mtcnn'):
        if method == 'mtcnn':
            logger.debug('using pre-trained MTCNN face extrctor')
            self.detector = mtcnn.MTCNN()
        if method == 'hog':
            logger.debug('using hog face extractor')
            self.detector = FaceRecognitionWrapper(model='hog')
    
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