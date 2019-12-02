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

        hist1, bins1 = np.histogram(t_method1, bins=20, range=(-3,3))
        hist2, bins2 = np.histogram(t_method2, bins=20, range=(-3,3))
        
        print(target, method1, method2)
        plt.plot(bins1[:-1], hist1)
        plt.plot(bins2[:-1], hist2)
        plt.legend([method1, method2])
        #plt.show()
        plt.savefig(f"/mnt/svm/code/Fairness/vis/pair_noconstraint/{target}_{method1}_{method2}.png")
        plt.close()
    return

def compareAllPairs(records, keys):
    preProcesses = ["Nothing","DisparateImpactRemover","LFR","OptimPreproc","Reweighing"]
    inProcesses = ["PlainModel","AdversarialDebiasing","ARTClassifier","PrejudiceRemover"]
    postProcesses = ["Nothing","CalibratedEqOddsPostprocessing","EqOddsPostprocessing","RejectOptionClassification"]
    for methods in [preProcesses, inProcesses, postProcesses]:
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                pairCompasion(records, keys, methods[i], methods[j])      
    return

class drawCircles():
    def __init__(self):
        ax = plt.gca()
        ax.set_aspect(0.6)
        #fig, ax = plt.subplots()
        ax.set_xlim((-10,1000))
        ax.set_ylim((0,810*3))
        ax.set_axis_off()
        self.ax = ax

    def addCircles(self, xs, ys, ds):
        for x, y, d in zip(xs, ys, ds):
            self.addCircle(x, y, d)

    def addCircle(self, x, y, d):
        x, y, d = int(x), int(y), int(d)
        circle = plt.Circle((x, y), d, color='k', fill=True)
        self.ax.add_artist(circle)
    
    def show(self):
        plt.show()

    def save(self, saveto):
        plt.savefig(saveto, dpi=300)
        plt.close()


def compareAll(records, keys):
    for target in ['TPR','TNR','FPR','FNR','Balanced_Acc','Acc',
            "Statistical parity difference","Disparate impact","Equal opportunity difference",
            "Average odds difference","Theil index","United Fairness"]:
        methodsAll = {}
        target_index = keys.index(target)
        
        ins = drawCircles()

        for record in records:
            methodsComb = str(record[:13])
            methodsAll[methodsComb] = methodsAll.get(methodsComb, []) + [record[target_index]]

        for index, combRes in enumerate(methodsAll.values()):
            minv = np.min(combRes)
            maxv = np.max(combRes)
            hist, bins = np.histogram(combRes, bins=80, range=(minv,maxv))
            bins = bins[:-1]
            bins = (bins - minv) * (1000/(maxv - minv))
            hist = (hist*6)/np.max(hist)
            ins.addCircles(bins, np.array([(index+1)*30]*80), hist)
        print(len(list(methodsAll.keys())))
        saveto = f"/mnt/svm/code/Fairness/vis/all/{target}.png"
        ins.save(saveto)
    return

if __name__ == "__main__":
    records, keys = getData()
    # compareAllPairs(records, keys)
    compareAll(records, keys)