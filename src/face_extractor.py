import face_recognition
from abc import ABC, abstractmethod


class FaceExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def getFaceLocations(self):
        pass
