from abc import ABC, abstractmethod

class ImperfectionRecognizer():

    def __init__(self, crackRecognizer, knotRecognizer):
        self.crackRecognizer = crackRecognizer
        self.knotRecognizer = knotRecognizer
        self.bboxList = []

    def getsAllImperfections(self, image):
        temp1 = self.crackRecognizer.imperfectionDetector(image)
        temp2 = self.knotRecognizer.imperfectionDetector(image)
        return temp1 + temp2

class AbstractRecognizer(ABC):

    @abstractmethod
    def imperfectionDetector(self):
        pass

    @abstractmethod
    def imperfectionSegmentator(self):
        pass

    @abstractmethod
    def imperfectionCuantification(self):
        pass


class CracksRecognizer(ImperfectionRecognizer):

    def __init__(self):
        self.bboxList = []

    def imperfectionDetector(self, binarizeImage):
        return 'CrackRecognizer imperfectionDetector'

    def imperfectionSegmentator(self, image):
        return 'CrackRecognizer imperfectionSegmentator'

    def imperfectionCuantification(self):
        return 'CrackRecognizer imperfectionCuantification'


class KnotsRecognizer(ImperfectionRecognizer):

    def __init__(self):
        self.bboxList = []

    def imperfectionDetector(self, binarizeImage):
        return 'KnotsRecognizer imperfectionDetector'

    def imperfectionSegmentator(self, image):
        return 'KnotsRecognizer imperfectionSegmentator'

    def imperfectionCuantification(self):
        return 'KnotsRecognizer imperfectionCuantification'


class BoundingBox():

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def getBbox(self):
        return 'Bbox'

    def getArea(self):
        return 'Area de Bbox'