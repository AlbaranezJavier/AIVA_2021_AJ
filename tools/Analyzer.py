from abc import ABC, abstractmethod
import numpy as np

class AbstractAnalyzer(ABC):

    @abstractmethod
    def do_quantification(self, quantifiable, non_quantifiable):
        pass

class MaskQuantificator(AbstractAnalyzer):

    def do_quantification(self, quantifiable, non_quantifiable):
        """
        Performs the quantization of the values of a mask, disregarding the non-quantifiable area.
        :param quantifiable: mask with the values to be quantified
        :param non_quantifiable: mask with the non-quantifiable values of the total
        :return: float
        """
        _absolute_total = quantifiable.shape[0]*quantifiable.shape[1]
        _relative_total = _absolute_total - np.sum(non_quantifiable)
        return (np.sum(quantifiable) / _relative_total) * 100
