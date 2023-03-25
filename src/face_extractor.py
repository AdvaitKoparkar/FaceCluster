import mtcnn

class FaceExtractor(object):
    def __init__(self, method = 'mtcnn'):
        if method == 'mtcnn':
            self.detector = mtcnn.MTCNN()
    
    def get_faces(self, img):
        face_locs = self.detector.detect_faces(img)
        faces, scores = [], []
        for face_loc in face_locs:
            faces.append(face_loc['box'])
            scores.append(face_loc['confidence'])
        return faces, scores
