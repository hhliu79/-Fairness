import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_txt_haipei, listDir
import os
import pickle


def getData():
    dirName = "/mnt/svm/code/Fairness/Haipei"
    
    if os.path.isfile(os.path.join(dirName, "records.pkl")):
        with open( os.path.join(dirName, "records.pkl"), 'rb' ) as f:
            return pickle.load(f)
    fileList = listDir(dirName, ends=".txt")
    records = None
    for fileName in fileList:
        instance, keys= load_txt_haipei(fileName)
        if records is None:
            records = instance
        else:
            records = np.concatenate((records, instance), axis=0)
    with open( os.path.join(dirName, "records.pkl"), 'wb' ) as f:
        pickle.dump((records,keys), f)
    return (records, keys)

def pairCompasion(records, keys, method1, method2):
    for target in ['TPR','TNR','FPR','FNR','Balanced_Acc','Acc',
            "Statistical parity difference","Disparate impact","Equal opportunity difference",
            "Average odds difference","Theil index","United Fairness"]:
        target_index = keys.index(target)
        target_v = records[:, target_index]
        
        method1_index = keys.index(method1)
        method2_index = keys.index(method2)
        method1_v = records[:, method1_index]
        method2_v = records[:, method2_index]
        
        row1 = np.where(method1_v == 1)[0]
        row2 = np.where(method2_v == 1)[0]

        t_method1 = target_v[row1]
        t_method2 = target_v[row2]

        plt.plot(t_method1)
        plt.plot(t_method2)
        plt.show()

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