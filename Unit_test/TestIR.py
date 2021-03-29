import unittest
from tools.ImperfectionRecognizer import *


class TestImperfectionRecognizer(unittest.TestCase):

    def test_imperfectionDetector(self):

        imagen = 'imagen'
        cr = CracksRecognizer()
        kr = KnotsRecognizer()
        ir = ImperfectionRecognizer(cr, kr)
        result = ir.getsAllImperfections(imagen)
        self.assertEqual(result,'CrackRecognizer imperfectionDetectorKnotsRecognizer imperfectionDetector')

    def test_crackRecognizer_imperfectionSegmentator(self):
        imagen = 'imagen'
        cr = CracksRecognizer()
        result = cr.imperfectionSegmentator(imagen)
        self.assertEqual(result, 'This has to fail!!')

    def test_knotRecognizer_imperfectionSegmentator(self):
        imagen = 'imagen'
        kr = KnotsRecognizer()
        result = kr.imperfectionSegmentator(imagen)
        self.assertEqual(result, 'KnotsRecognizer imperfectionSegmentator')

    def test_crackRecognizer_imperfectionCuantification(self):
        cr = CracksRecognizer()
        result = cr.imperfectionCuantification()
        self.assertEqual(result, 'CrackRecognizer imperfectionCuantification')

    def test_knotRecognizer_imperfectionCuantification(self):
        kr = KnotsRecognizer()
        result = kr.imperfectionCuantification()
        self.assertEqual(result, 'KnotsRecognizer imperfectionCuantification')

if __name__ == '__main__':
    unittest.main()