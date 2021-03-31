import unittest
from TestSD import TestStatistics
from TestIR import TestImperfectionRecognizer

"""
This script executes a set of tests
"""

class SuiteTest():

    def run(self):
        """
        Run the TestStatistics and TestImperfectionRecognizer tests.
        :return: results
        """
        suite = unittest.TestSuite()
        suite.addTest(TestStatistics("test_statistics"))
        suite.addTest(TestImperfectionRecognizer("test_comparation"))
        suite.addTest(TestImperfectionRecognizer("test_binary_mask"))

        runner = unittest.TextTestRunner()
        runner.run(suite)


if __name__ == '__main__':
    st = SuiteTest()
    st.run()
