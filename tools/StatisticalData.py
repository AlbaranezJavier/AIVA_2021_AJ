from tabulate import tabulate
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import f

"""
This script contains a code that analyzes a method showing a table of statistics.
"""


class Statistics:

    def __init__(self, p):
        self.confidence_interval_p = p
        self.mydict = {
            "population": 0, "pp": 0, "bias": 0, "pn": 0, "ib": 0, "rp": 0, "tp": 0, "recall": 0, "fn": 0, "fnr": 0,
            "lr+": 0, "prevalence": 0, "precision": 0, "performance": 0, "fna": 0, "irr": 0, "lr-": 0, "rn": 0, "fp": 0,
            "fpr": 0, "tn": 0, "specifity": 0, "dor": 0, "ner": 0, "fdr": 0, "der": 0, "ip": 0, "crr": 0,
            "informedness": 0, "chi": 0, "correlation": 0, "pra": 0, "markedness": 0, "accuracy": 0, "mcc": 0, "iou": 0,
            "ck": 0, "mr": 0, "f1": 0,
            "i_recall": [0, 0], "i_fnr": 0, "i_precision": 0, "i_performance": 0, "i_fna": 0,
            "i_irr": 0, "i_bias": 0, "i_ib": 0, "i_prevalence": 0, "i_fpr": 0, "i_tnr": 0, "i_ner": 0, "i_fdr": 0,
            "i_der": 0, "i_ip": 0, "i_crr": 0, "i_correlation": 0, "i_accuracy": 0, "i_iou": 0, "i_mr": 0, "i_f1": 0,
            "i_specifity": 0, "p": 0
        }

    def tp_fn_fp_tn(self, predicted, gt):
        """
        Calculates tp, fn, fp and tn for a predicted mask and its ground thruth
        :param predicted: mask
        :param gt: mask
        :return: tp, fn, fp, tn
        """
        # variables
        _ones = np.ones_like(predicted)
        _predicted_zeros = np.logical_not(predicted)
        _gt_zeros = np.logical_not(gt)

        # TP, FN, FP, TN
        self.mydict["tp"] = np.sum(np.logical_and(predicted, gt))
        self.mydict["fn"] = np.sum(np.logical_and(_predicted_zeros, gt))
        self.mydict["fp"] = np.sum(np.logical_and(predicted, _gt_zeros))
        self.mydict["tn"] = np.sum(np.logical_and(_predicted_zeros, _gt_zeros))

        return self.mydict["tp"], self.mydict["fn"], self.mydict["fp"], self.mydict["tn"]

    def test(self, tp, fn, fp, tn):
        """
        Initializes some values for tp, fn, fp, tn
        :param tp: true positive
        :param fn: false negative
        :param fp: false positive
        :param tn: true negative
        :return:
        """
        # TP, FN, FP, TN
        self.mydict["tp"] = tp
        self.mydict["fn"] = fn
        self.mydict["fp"] = fp
        self.mydict["tn"] = tn

    def others(self):
        """
        Calculates the metrics of:
        Population, Predicted Positive, Predicted Negative, Real Positive, Real Negative
        Bias, Inverse Bias, Prevalence, Null Error Rate
        Recall, FNR, FPR, FNR,
        Performance, Incorrect Rejection rate, Delivered Error rate, Correct Rejection Rate
        Precision, False Negative Rate, False Discovery Rate, Inverse Precision
        Accuracy
        Informedness, Markedness
        LR+, LR-, Diagnostic Odds Ratio
        Correlation
        Probability of random agreement, Matthews corr. coeff., IoU, Cohen's Kappa, Misclasification rate, f1 score
        :return: None
        """
        # Population, Predicted Positive, Predicted Negative, Real Positive, Real Negative
        self.mydict["population"] = self.mydict["tp"] + self.mydict["tn"] + self.mydict["fp"] + self.mydict["fn"]
        self.mydict["pp"] = self.mydict["tp"] + self.mydict["fp"]
        self.mydict["pn"] = self.mydict["fn"] + self.mydict["tn"]
        self.mydict["rp"] = self.mydict["tp"] + self.mydict["fn"]
        self.mydict["rn"] = self.mydict["fp"] + self.mydict["tn"]

        # Bias, Inverse Bias, Prevalence, Null Error Rate
        self.mydict["bias"] = np.around((self.mydict["pp"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_bias"] = np.around(proportion_confint(count=self.mydict["pp"], nobs=self.mydict["population"],
                                                             alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["ib"] = np.around((self.mydict["pn"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_ib"] = np.around(
            proportion_confint(count=self.mydict["pn"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["prevalence"] = np.around((self.mydict["rp"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_prevalence"] = np.around(
            proportion_confint(count=self.mydict["rp"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["ner"] = np.around((self.mydict["rn"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_ner"] = np.around(
            proportion_confint(count=self.mydict["rn"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100

        # Recall, FNR, FPR, FNR,
        self.mydict["recall"] = np.around((self.mydict["tp"] / self.mydict["rp"]) * 100, 2)
        self.mydict["i_recall"] = np.around(
            proportion_confint(count=self.mydict["tp"], nobs=self.mydict["rp"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["fnr"] = np.around((self.mydict["fn"] / self.mydict["rp"]) * 100, 2)
        self.mydict["i_fnr"] = np.around(
            proportion_confint(count=self.mydict["fn"], nobs=self.mydict["rp"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["fpr"] = np.around((self.mydict["fp"] / self.mydict["rn"]) * 100, 2)
        self.mydict["i_fpr"] = np.around(
            proportion_confint(count=self.mydict["fp"], nobs=self.mydict["rn"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["specifity"] = np.around((self.mydict["tn"] / self.mydict["rn"]) * 100, 2)
        self.mydict["i_specifity"] = np.around(
            proportion_confint(count=self.mydict["tn"], nobs=self.mydict["rn"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100

        # Performance, Incorrect Rejection rate, Delivered Error rate, Correct Rejection Rate
        self.mydict["performance"] = np.around((self.mydict["tp"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_performance"] = np.around(
            proportion_confint(count=self.mydict["tp"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["irr"] = np.around((self.mydict["fn"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_irr"] = np.around(
            proportion_confint(count=self.mydict["fn"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["der"] = np.around((self.mydict["fp"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_der"] = np.around(
            proportion_confint(count=self.mydict["fp"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["crr"] = np.around((self.mydict["tn"] / self.mydict["population"]) * 100, 2)
        self.mydict["i_crr"] = np.around(
            proportion_confint(count=self.mydict["tn"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100

        #  Precision, False Negative Rate, False Discovery Rate, Inverse Precision
        self.mydict["precision"] = np.around((self.mydict["tp"] / self.mydict["pp"]) * 100, 2)
        self.mydict["i_precision"] = np.around(
            proportion_confint(count=self.mydict["tp"], nobs=self.mydict["pp"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["fna"] = np.around((self.mydict["fn"] / self.mydict["pn"]) * 100, 2)
        self.mydict["i_fna"] = np.around(
            proportion_confint(count=self.mydict["fn"], nobs=self.mydict["pn"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["fdr"] = np.around((self.mydict["fp"] / self.mydict["pp"]) * 100, 2)
        self.mydict["i_fdr"] = np.around(
            proportion_confint(count=self.mydict["fp"], nobs=self.mydict["pp"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["ip"] = np.around((self.mydict["tn"] / self.mydict["pn"]) * 100, 2)
        self.mydict["i_ip"] = np.around(
            proportion_confint(count=self.mydict["tn"], nobs=self.mydict["pn"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100

        # Accuracy
        self.mydict["accuracy"] = np.around(
            ((self.mydict["tp"] + self.mydict["tn"]) / self.mydict["population"]) * 100, 2)
        self.mydict["i_accuracy"] = np.around(
            proportion_confint(count=(self.mydict["tp"] + self.mydict["tn"]),
                               nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100

        # Informedness, Markedness
        self.mydict["informedness"] = np.around((self.mydict["recall"] - self.mydict["fpr"]), 2)
        self.mydict["markedness"] = np.around((self.mydict["precision"] - self.mydict["fna"]), 2)

        # LR+, LR-, Diagnostic Odds Ratio
        self.mydict["lr+"] = np.around((self.mydict["recall"] / self.mydict["fpr"]), 2)
        self.mydict["lr-"] = np.around((self.mydict["fnr"] / self.mydict["specifity"]), 2)
        self.mydict["dor"] = np.around((self.mydict["lr+"] / self.mydict["lr-"]), 2)

        # Chi square
        _temp11 = (self.mydict["rp"] * self.mydict["pp"]) / self.mydict["population"]
        _temp12 = (self.mydict["rp"] * self.mydict["pn"]) / self.mydict["population"]
        _temp21 = (self.mydict["rn"] * self.mydict["pp"]) / self.mydict["population"]
        _temp22 = (self.mydict["rn"] * self.mydict["pn"]) / self.mydict["population"]

        _temp11 = (self.mydict["tp"] - _temp11) ** 2 / _temp11
        _temp12 = (self.mydict["fp"] - _temp12) ** 2 / _temp12
        _temp21 = (self.mydict["fn"] - _temp21) ** 2 / _temp21
        _temp22 = (self.mydict["tn"] - _temp22) ** 2 / _temp22

        self.mydict["chi"] = np.around(_temp11 + _temp12 + _temp21 + _temp22, 2)
        self.mydict["p"] = np.around(1 - f.cdf(self.mydict["chi"], 1, 1000000000), 2)

        # Correlation
        self.mydict["correlation"] = np.around(
            (self.mydict["tp"] * self.mydict["tn"] + self.mydict["fp"] * self.mydict["fn"]) / (
                    self.mydict["pp"] * self.mydict["rp"] * self.mydict["rn"] * self.mydict[
                "pn"]) ** .5, 2)
        self.mydict["i_correlation"] = np.around(
            proportion_confint(
                count=(self.mydict["tp"] * self.mydict["tn"] + self.mydict["fp"] * self.mydict["fn"]),
                nobs=(self.mydict["pp"] * self.mydict["rp"] * self.mydict["rn"] * self.mydict[
                    "pn"]) ** .5,
                alpha=self.confidence_interval_p, method="beta"), 2)

        # Probability of random agreement, Matthews corr. coeff., IoU, Cohen's Kappa, Misclasification rate, f1 score
        self.mydict["pra"] = np.around(
            (self.mydict["pp"] * self.mydict["rp"] + self.mydict["pn"] * self.mydict["rn"]) /
            self.mydict["population"] ** 2, 2) * 100
        self.mydict["mcc"] = np.around((self.mydict["chi"] / self.mydict["population"]) ** 0.5, 2)
        self.mydict["ck"] = np.around(
            (self.mydict["accuracy"] - self.mydict["pra"]) / (100 - self.mydict["pra"]), 2)
        self.mydict["mr"] = np.around(
            (self.mydict["fp"] + self.mydict["fn"]) / self.mydict["population"], 2) * 100
        self.mydict["i_mr"] = np.around(
            proportion_confint(count=self.mydict["fp"] + self.mydict["fn"], nobs=self.mydict["population"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["iou"] = np.around(
            self.mydict["tp"] / (self.mydict["population"] - self.mydict["tn"]), 2) * 100
        self.mydict["i_iou"] = np.around(
            proportion_confint(count=self.mydict["tp"], nobs=self.mydict["population"] - self.mydict["tn"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100
        self.mydict["f1"] = np.around(2 * self.mydict["tp"] / (self.mydict["rp"] + self.mydict["pp"]),
                                      2)
        self.mydict["i_f1"] = np.around(
            proportion_confint(count=2 * self.mydict["tp"], nobs=self.mydict["rp"] + self.mydict["pp"],
                               alpha=self.confidence_interval_p, method="beta"), 2) * 100

    def print_table(self):
        """
        Draws a table with all the metrics
        :return:
        """
        # Creating table
        table = [[f'Population\nN = TP+TN+FP+FN\n{self.mydict["population"]}', "", f'Predicted Positive\nPP = TP+FP\n{self.mydict["pp"]}',
                  f'Bias\npp = PP/N\n{self.mydict["bias"]} {self.mydict["i_bias"]}%', f'Predicted Negative\nPN = FN+TN\n{self.mydict["pn"]}',
                  f'Inverse Bias\npn = PN/N\n{self.mydict["ib"]} {self.mydict["i_ib"]}%', "", ""],
                 [],
                 [f'Real Positive\nRP = TP+FN\n{self.mydict["rp"]}', "", f'TP\n\n{self.mydict["tp"]}',
                  f'Recall\ntpr = TP/RP\n{self.mydict["recall"]} {self.mydict["i_recall"]}%',
                  f'FN\n\n{self.mydict["fn"]}', f'FNR\nfnr = FN/RP\n{self.mydict["fnr"]} {self.mydict["i_fnr"]}%', "",
                  f'LR+\nLR+ = tpr/fpr\n{self.mydict["lr+"]}'],
                 [f'Prevalence\nrp = RP/N\n{self.mydict["prevalence"]} {self.mydict["i_prevalence"]}%', "",
                  f'Precision\ntpa = TP/PP\n{self.mydict["precision"]} {self.mydict["i_precision"]}%',
                  f'Performance\ntp = TP/N\n{self.mydict["performance"]} {self.mydict["i_performance"]}%',
                  f'FN Accuracy\nfna = FN/PN\n{self.mydict["fna"]} {self.mydict["i_fna"]}%',
                  f'Incorrect Rejection Rate\nfn = FN/N\n{self.mydict["irr"]} {self.mydict["i_irr"]}%', "",
                  f'LR-\nLR- = fnr/tnr\n{self.mydict["lr-"]}'],
                 [],
                 [f'Real Negative\nRN = FP+TN\n{self.mydict["rn"]}', "", f'FP\n\n{self.mydict["fp"]}',
                  f'FPR\nfpr = FP/RN\n{self.mydict["fpr"]} {self.mydict["i_fpr"]}%',
                  f'TN\n\n{self.mydict["tn"]}', f'Specifity\ntnr = TN/RN\n{self.mydict["specifity"]} {self.mydict["i_specifity"]}%',
                  "",
                  f'Odds ratio\ndor = LR+/LR-\n{self.mydict["dor"]}'],
                 [f'Null Error Rate\nrn = RN/N\n{self.mydict["ner"]} {self.mydict["i_ner"]}%', "",
                  f'False Discovery Rate\nfdr = FP/PP\n{self.mydict["fdr"]} {self.mydict["i_fdr"]}%',
                  f'Delivered Error Rate\nfp = FP/N\n{self.mydict["der"]} {self.mydict["i_der"]}%',
                  f'Inverse Precision\ntna = TN/PN\n{self.mydict["ip"]} {self.mydict["i_ip"]}%',
                  f'Correct Rejection Rate\ntn = TN/N\n{self.mydict["crr"]} {self.mydict["i_crr"]}%', "",
                  f'Informedness\n = tpr - fpr\n{self.mydict["informedness"]}%'],
                 [],
                 ["", "", f'Chi square\n{self.mydict["chi"]}\np={self.mydict["p"]}',
                  f'Correlation\n(TP*TN - FP*FN)/(PP*RP*RN*PN)**.5\n{self.mydict["correlation"]} {self.mydict["i_correlation"]}',
                  f'Prob. Random Agreement\npra = (PP*RP + PN*RN)/(N)**2\n{self.mydict["pra"]}%', f'Markedness\n= tpa - fna\n{self.mydict["markedness"]}%', "",
                  f'Accuracy\nacc = (TP + TN) / N\n{self.mydict["accuracy"]} {self.mydict["i_accuracy"]}%'],
                 ["", "", f'Matthews Corr. Coeff.\nmcc = (chi / N)**.5\n{self.mydict["mcc"]}',
                  f'IoU\n= TP/(N-TN)\n{self.mydict["iou"]} {self.mydict["i_iou"]}%',
                  f'Cohen Kappa\n= (acc-pra)/(1-pra)\n{self.mydict["ck"]}',
                  f'Misclassification Rate\nerr = (PF+FN)/N\n{self.mydict["mr"]} {self.mydict["i_mr"]}%', "",
                  f'F1 score\n= 2*TP/(RP+PP)\n{self.mydict["f1"]} {self.mydict["i_f1"]}%']]

        # Displaying table
        print(tabulate(table, tablefmt='grid',
                       colalign=("center", "center", "center", "center", "center", "center", "center", "center")))


if __name__ == '__main__':
    s = Statistics(p=0.01)
    s.test(50, 20, 10, 200)
    s.others()
    s.print_table()
