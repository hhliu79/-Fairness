import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_txt_haipei, listDir


def getData():
    dirName = "/mnt/svm/code/Fairness/Haipei"
    fileList = listDir(dirName, ends=".txt")
    records = None
    for fileName in fileList:
        instance, keys= load_txt_haipei(fileName)
        if records is None:
            records = instance
        else:
            records = np.concatenate((records, instance), axis=0)
    return records, keys

def pairCompasion(records, keys, method1, method2):
    return

def compareAll(records, keys):
    preProcesses = ["Nothing","DisparateImpactRemover","LFR","OptimPreproc","Reweighing"]
    inProcesses = ["PlainModel","AdversarialDebiasing","ARTClassifier","PrejudiceRemover"]
    postProcesses = ["Nothing","CalibratedEqOddsPostprocessing","EqOddsPostprocessing","RejectOptionClassification"]
    for methods in [preProcesses, inProcesses, postProcesses]:
        for i in range(len(methods)):
            for j in range(i, len(methods)):
                pairCompasion(records, keys, methods[i], methods[j])      
    return

if __name__ == "__main__":
    records, keys = getData()
    compareAll(records, keys)