"""
file for compare model
"""

from data_utils import load_txt_haipei, listDir
import numpy as np
import os
import pickle
import itertools
import tqdm

targetMap = {"B_acc": "Balanced_Acc",
                "acc": "Acc",
                "SPD": "Statistical parity difference",
                "DIC": "Disparate impact",
                "EOD": "Equal opportunity difference",
                "AOD": "Average odds difference",
                "TI": "Theil index",
                "UF": "United Fairness"}

def write2file(f, records, keys, target):
    """
    args:
        - target: [acc, B_acc, SPD, DIC, EOD, AOD, TI, UF]
            ['Balanced_Acc','Acc',
            "Statistical parity difference","Disparate impact","Equal opportunity difference",
            "Average odds difference","Theil index","United Fairness"]
        - records: batch records with same dataset attribute
        - f: file to write records
    """
    perms = itertools.combinations(range(records.shape[0]), 2)
    targetIndex = keys.index(targetMap[target])
    for perm in perms:
        perm = np.array(perm)
        np.random.shuffle(perm)
        records0 = records[perm[0]]
        records1 = records[perm[1]]
        feature = np.concatenate( (records0[0:0+13], records1[0:0+13], records0[13:13+23]) )
        label = 1 if records0[targetIndex] > records1[targetIndex] else 0
        f.write(f"{label}")
        for indice, value in enumerate(feature):
            f.write(f" {indice}:{value}")
        f.write("\n")
    
    return None

def getCombPairs(records, keys):
    perms = itertools.combinations(range(records.shape[0]), 2)
    features = []
    labels = []
    for perm in perms:
        perm = np.array(perm)
        np.random.shuffle(perm)
        records0 = records[perm[0]]
        records1 = records[perm[1]]
        features.append( np.concatenate( (records0[0:0+13], records1[0:0+13], records0[13:13+23]) ).reshape((1,-1)) )
        label = (records0[40:40+8] - records1[40:40+8]).reshape((1,-1))
        labels.append( label )
    features = np.concatenate(tuple(features), axis=0)
    labels = np.concatenate(tuple(labels), axis=0)
    
    return (features, labels)

def pairs2txt(pairs, params):
    """
    args:
        paris: {fileName: (features, labels)} labels include all the eight in the float format (<0 or >0)
    
    save to txt for xgboost
    """
    return 0

def loadParial(params):
    saveto = params['saveto']
    allFile = os.path.join(saveto, 'pairwisedata.pkl')
    partialFile = os.path.join(saveto, 'pairwisedataPartial.pkl')
    
    if os.path.isfile(partialFile):
        with open(partialFile, 'rb') as f:
            return pickle.load(f)
    

    with open(allFile, 'rb') as f:
        allPairs = pickle.load(f)
    
    partialPairs = {}
    for fileName in allPairs.keys():
        features, labels = allPairs[fileName]
        print(features.shape, labels.shape)
        assert features.shape[0] == labels.shape[0]
        indexes = list(range(features.shape[0]))
        np.random.shuffle(indexes)
        features = features[indexes, :]
        labels = labels[indexes, :]
        sampleN = min(params['sampleN'], labels.shape[0])
        partialPairs[fileName] = (features[:sampleN,:], labels[:sampleN, :])
    
    with open(partialFile, 'wb') as f:
        pickle.dump(partialPairs, f)


def dict2pairs(records_seperate, params):
    """
    for each key (fileName) of records_seperate, generate a xgboost format txt data
    records:
        0:0+13:     method
        13:13+23:   dataset attribute
        40:40+8:    targets 
    return :
        dict: filename: pair record for all target
    """
    saveto = params['saveto']
    savetoFile = os.path.join(saveto, 'pairwisedata.pkl')
    
    if os.path.isfile(savetoFile):
        with open(savetoFile, 'rb') as f:
            return pickle.load(f)

    pairs = {}  
    for fileName in records_seperate:
        print('process: ', fileName)
        records, keys = records_seperate[fileName]
        featuresPerFile = []
        labelsPerFile = []
        
        prevAttr = None
        batchIndex = []
        for index in tqdm.tqdm(range(records.shape[0])):
            curAttr = str(tuple(records[index, 13:13+23]))
            if prevAttr is not None and prevAttr != curAttr:
                """process a batch"""
                if len(batchIndex) < 2:
                    print("!! only one record with the same subdataset !!")
                else:
                    batchRecord = records[batchIndex]
                    features, labels = getCombPairs(batchRecord, keys)
                    featuresPerFile.append(features)
                    labelsPerFile.append(labels)
                batchIndex = [index]
            else:
                batchIndex.append(index)
            prevAttr = curAttr
        featuresPerFile = np.concatenate(tuple(featuresPerFile), axis=0)
        labelsPerFile = np.concatenate(tuple(labelsPerFile), axis=0)
        pairs[fileName] = (featuresPerFile, labelsPerFile)

    savetoFile = os.path.join(saveto, 'pairwisedata.pkl')
    with open(savetoFile, 'wb') as f:
        pickle.dump(pairs, f)

def dict2dmatrix(records_seperate, params):
    """
    for each key (fileName) of records_seperate, generate a xgboost format txt data
    records:
        0:0+13:     method
        13:13+23:   dataset attribute
        40:40+8:    targets 
    """
    target = params['target']
    saveto = params['saveto']
    for fileName in records_seperate:
        print("process: ", fileName)
        with open(os.path.join(saveto, fileName.split('.txt')[0]+target+'.txt'), 'w') as f:
            
            records, keys = records_seperate[fileName]
            
            prevAttr = None
            batchIndex = []
            for index in tqdm.tqdm(range(records.shape[0])):
                curAttr = str(tuple(records[index, 13:13+23]))
                if prevAttr is not None and prevAttr != curAttr:
                    """process a batch"""
                    write2file(f, records[batchIndex], keys, target)
                    batchIndex = []
                else:
                    batchIndex.append(index)
                prevAttr = curAttr


def txt2dict(dataDir=None):
    """
    return:
        {fileName: 
                (records: -> numpy array
                keys:    -> list['str'])
        }
    """
    dataDir = "/mnt/svm/code/Fairness/Haipei_80" if not dataDir else dataDir

    dictFile = os.path.join(dataDir, 'records_seperate.pkl')
    if os.path.isfile(dictFile):
        with open(dictFile, 'rb') as f:
            return pickle.load(f)

    fileList = listDir(dataDir, ends=".txt")
    records_seperate = {}
    records = None
    for fileName in fileList:
        baseName = os.path.basename(fileName)
        instance, keys= load_txt_haipei(fileName)
        if records is None:
            records = instance
        else:
            records = np.concatenate((records, instance), axis=0)
        records_seperate[baseName] = (instance, keys)
    with open(dictFile, 'wb') as f:
        pickle.dump(records_seperate, f)

    return records_seperate


def prepareTxt(dictData, params):
    """
    files : ['vectors_adult_race_.txt', 'vectors_adult_sex_.txt', 'vectors_compas_race_.txt', 'vectors_compas_sex_.txt']
    labels : ["B_acc", "acc", "SPD", "DIC", "EOD", "AOD", "TI", "UF"]
    params:
        
    """
    files = ['vectors_adult_race_.txt', 'vectors_adult_sex_.txt', 'vectors_compas_race_.txt', 'vectors_compas_sex_.txt']
    labelNames = ["B_acc", "acc", "SPD", "DIC", "EOD", "AOD", "TI", "UF"]
    labelIndex = labelNames.index(params['target'])
    mode = params['mode']
    saveto = params['saveto']
    trainRatio = params['trainRatio']
    featSelect = params['featSelect'] if 'featSelect' in params else None
    
    trainFile = os.path.join(saveto, 'xgboostTrain.txt')
    testFile = os.path.join(saveto, 'xgboostTest.txt')
    if mode in files:
        allfeatures, alllabels = dictData[mode]
        if featSelect is not None:
            allfeatures = allfeatures[:, featSelect] 
            
    if mode == 'mixup':
        allfeatures = []
        alllabels = []
        for fileName in dictData:
            features, labels = dictData[fileName]
            if featSelect is not None:
                features = features[:, featSelect] 
            allfeatures.append(features)
            alllabels.append(labels)
        allfeatures = np.concatenate(tuple(allfeatures), axis=0)
        alllabels = np.concatenate(tuple(alllabels), axis=0)
        
        
    sampleN = allfeatures.shape[0]
    indexes = list(range(sampleN))
    np.random.shuffle(indexes)
    trainIndex = indexes[:int(trainRatio*sampleN)]
    testIndex = indexes[int(trainRatio*sampleN):]
    
    f = open(trainFile, 'w')
    for feat, label in zip(allfeatures[trainIndex], alllabels[trainIndex]):
        label = label[labelIndex]
        label = 1 if label>=0 else 0
        f.write(f"{label}")
        for indice, value in enumerate(feat):
            f.write(f" {indice}:{value}")
        f.write("\n")
    f.close()

    f = open(testFile, 'w')
    for feat, label in zip(allfeatures[testIndex], alllabels[testIndex]):
        label = label[labelIndex]
        label = 1 if label>=0 else 0
        f.write(f"{label}")
        for indice, value in enumerate(feat):
            f.write(f" {indice}:{value}")
        f.write("\n")
    f.close()

from data_utils import class_to_onehot
def dataFeat2CombMatrix(dataFeats, params, methods=[5, 4, 4]):
    """
    args:
        dataFeats: numpy array of datsFeats, each row is a record
        methods: the numbers of each methods (pre, in, post)
        params: saveto
    return:
        all permutation of method and their features
    """
    label = 1.0
    f = params['saveto']
    
    with open(f, 'w') as f:    
        for feature in dataFeats:
            for (preMethod, inMethod, postMethod) in itertools.product(range(5), range(4), range(4)):
                method1Feat = class_to_onehot([preMethod, inMethod, postMethod], {'methods':[5,4,4]})
                for (preMethod, inMethod, postMethod) in itertools.product(range(5), range(4), range(4)):
                    method2Feat = class_to_onehot([preMethod, inMethod, postMethod], {'methods':[5,4,4]})
                    f.write(f"{label}")
                    methodDataFeature = np.concatenate((method1Feat, method2Feat, feature), axis=0)
                    for indice, value in enumerate(methodDataFeature):
                        f.write(f" {indice}:{value}")
                    f.write("\n")

if __name__ == "__main__":
    """
    # test dataFeat2CombMatrix 
    dataFeats = np.zeros((2,23))
    params = {}
    params['saveto'] = './combination.txt'
    dataFeat2CombMatrix(dataFeats, params)
    """

    """
    records_seperate = txt2dict()
    print(records_seperate.keys())
    params = {'saveto': "/mnt/svm/code/Fairness/Haipei_80", 'sampleN': 100000}
    # dict2pairs(records_seperate, params)
    dictData = loadParial(params)
    params = {  'target':'acc', 
                'mode':'mixup', 
                'saveto': "/mnt/svm/code/Fairness/Haipei_80/xgboostFormat",
                'trainRatio': 0.7
            }
    prepareTxt(dictData, params)
    """
    """
    for target in ["B_acc", "acc", "SPD", "DIC", "EOD", "AOD", "TI", "UF"]:
        params = {'saveto': "/mnt/svm/code/Fairness/Haipei_80/xgboostFormat",
              'target': target  }
        dict2dmatrix(records_seperate, params)
    """