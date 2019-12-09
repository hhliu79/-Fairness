import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_txt_haipei, listDir
import os
import pickle
import tqdm

def getData(fileName=None):
    if fileName:
        with open( fileName, 'rb' ) as f:
            return pickle.load(f)
   
    dirName = "/mnt/svm/code/Fairness/Haipei"
 
    fileList = listDir(dirName, ends=".txt")
    records = None
    for fileName in fileList:
        instance, keys= load_txt_haipei(fileName)
        saveto = fileName.replace('.txt', '.pkl')
        assert saveto!=fileName
        with open(saveto, 'wb') as f:
            pickle.dump((instance, keys), f)

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

def ranking(values):
    rank = np.argsort(values)
    return rank

def aggregateRankings(loadfrom=None):
    targets = ['Balanced_Acc','Acc', "Statistical parity difference","Disparate impact",
                "Equal opportunity difference","Average odds difference","Theil index","United Fairness"]
    targets_showName = ['BAcc','Acc', "SPD","DI",
                "EOD","AOD","TI","UF"]

    if loadfrom is None:
        loadfrom = f'/mnt/svm/code/Fairness/vis/ranks/ranks.pkl'
    with open(loadfrom, 'rb') as f:
        methodsAllTarget = pickle.load(f)

    ranksTargetsMethods = {}
    allComb = []
    # target->method->rank
    for target in methodsAllTarget.keys():
        ranksPerTarget = {}
        methodsPerTarget = methodsAllTarget[target]
        methods= []
        performs = []
        for method in methodsPerTarget.keys():
            meanPerformance = np.mean(methodsPerTarget[method])
            methods.append(method)
            performs.append(meanPerformance)
        ranks = ranking(performs)
        allComb = methods
        for method, rank in zip(methods, ranks):
            ranksPerTarget[method] = rank
        
        ranksTargetsMethods[target] = ranksPerTarget

    # for combIndex, comb in enumerate(sorted(allComb)):
    #     rankAllTargets = []
    #     for target in targets:
    #         rankAllTargets.append(ranksTargetsMethods[target][comb])
        
    #     plt.plot(np.array(rankAllTargets))
    #     plt.xticks(list(range(8)), targets_showName)
    #     title = 'dataset: ' + os.path.basename(loadfrom) + '\n' + 'methods: ' + comb + '\n' + f'methodsIndex: {combIndex}'
    #     saveName = 'dataset:' + os.path.basename(loadfrom) + 'methods:' + comb + f'methodsIndex:{combIndex}'
    #     saveName = "/mnt/svm/code/Fairness/vis/ranks/perMethodperDataset/"+saveName+'.png'
        
    #     plt.title(title)
    #     plt.ylabel('ranking')
    #     plt.xlabel('criteria')
    #     plt.subplots_adjust(top=0.83)
    #     plt.savefig(saveName)
    #     plt.close()
    
    # for target in targets:
    #     methodsPerTarget = []
    #     for combIndex, comb in enumerate(sorted(allComb)):
    #         methodsPerTarget.append(ranksTargetsMethods[target][comb])
        
    #     plt.bar(list(range(80)), np.array(methodsPerTarget))
    #     # plt.xticks(list(range(80)), list(range(80)))
    #     title = 'dataset: ' + os.path.basename(loadfrom) + '\n' + 'target: ' + target
    #     saveName = 'dataset:' + os.path.basename(loadfrom) + 'target:' + target
    #     saveName = "/mnt/svm/code/Fairness/vis/ranks/methodsPerTarget/"+saveName+'.png'
        
    #     plt.title(title)
    #     plt.ylabel('ranking')
    #     plt.xlabel('comb method index')
    #     plt.subplots_adjust(top=0.83)
    #     # plt.show()
    #     plt.savefig(saveName)
    #     plt.close()
    
    totalRanking = []
    for combIndex, comb in enumerate(sorted(allComb)):
        perMethod = 0
        for target in targets:
            perMethod += ranksTargetsMethods[target][comb]
        totalRanking.append(perMethod)
    
    plt.bar(list(range(80)), np.array(totalRanking))
    title = 'dataset: ' + os.path.basename(loadfrom)
    saveName = 'dataset:' + os.path.basename(loadfrom)
    saveName = "/mnt/svm/code/Fairness/vis/ranks/totalRankingPerDataset/"+saveName+'.png'
    
    plt.title(title)
    plt.ylabel('ranking')
    plt.xlabel('comb method index')
    plt.subplots_adjust(top=0.83)
    # plt.show()
    plt.savefig(saveName)
    plt.close()


def rankingAll(records, keys, fileName=None):
    methodsAllTarget = {}
    if fileName:
        saveto = fileName.replace('.pkl', '_ranks.pkl')
        assert saveto != fileName
    else:
        saveto = f'/mnt/svm/code/Fairness/vis/ranks/ranks.pkl'
    # 'TPR','TNR','FPR','FNR',
    targets = ['Balanced_Acc','Acc', "Statistical parity difference","Disparate impact",
                "Equal opportunity difference","Average odds difference","Theil index","United Fairness"]
    bests = [1, 1, 0, 1, 0, 0, 0, 0]
    for target, best in tqdm.tqdm(zip(targets, bests)):
        methodsPerTarget = {}
        target_index = keys.index(target)
    
        # compute all record according to the target and best
        for record in records:
            # for single record
            methodsComb = str(record[:13])
            diffFromBest = np.abs(record[target_index] - best) # smaller is better
            methodsPerTarget[methodsComb] = methodsPerTarget.get(methodsComb, []) + [diffFromBest]
        
        methodsAllTarget[target] = methodsPerTarget

    with open(saveto, 'wb') as f:
        pickle.dump(methodsAllTarget, f)
    return methodsAllTarget


if __name__ == "__main__":
    dirName = '/mnt/svm/code/Fairness/Haipei'
    fileList = listDir(dirName, ends=".pkl")
    for fileName in fileList:
        if '_ranks.pkl' in fileName:
            continue
        # records, keys = getData(fileName)
        # rankingAll(records, keys, fileName=fileName)
        
        loadfrom = fileName.replace('.pkl', '_ranks.pkl')
        aggregateRankings(loadfrom)
    
    # compareAllPairs(records, keys)
    # compareAll(records, keys)
    