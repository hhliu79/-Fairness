import csv
import numpy as np
import re
import os


def class_to_onehot(classList, params):
    """
    [1,0,2] -> [0,1,0, 1,0,0, 0,0,1 ]
    """
    feature = None
    for classes, nhot in zip(classList, params['methods']):
        clsFeat = np.zeros((nhot,))
        clsFeat[int(classes)] = 1.0
        if feature is None:
            feature = clsFeat
        else:
            feature = np.concatenate((feature, clsFeat), axis=0)
    return feature


def gen_sample(content, params):
    """
    generate one feature from content
    Returns:
        feature: np.array
    """
    classFeature = class_to_onehot(content[-3:], params)
    otherFeature = np.array(content[:-3])
    feature = np.concatenate((otherFeature, classFeature), axis=0)
    return feature


def load_csv(fileList, params={}):
    """
    load_csv from filePath
    labels are generated according to the target values
    Args:
        fileList: all files we want to load, they will be combined together and returned
        params: {
            "target": one of [acc, B_acc, SPD, DIC, EOD, AOD, TI, UF]
        }
    Returns:
        features: n*x
        labels: n*1
    """
    # get the params
    defaultParams = {'methods': [5,4,4], 'target':"UF"}
    for pkey in defaultParams.keys():
        if pkey not in params:
            params[pkey] = defaultParams[pkey]

    keys = None 
    all_samples = None
    for filePath in fileList:
        # get the length
        with open(filePath) as f:
            csvReader = csv.reader(f, delimiter=',')
            lengthSamples = sum(1 for _ in csvReader)
            lengthSamples -= 1

        with open(filePath) as f:
            csvReader = csv.reader(f, delimiter=',')
            for row, content in enumerate(csvReader):
                if row == 0:
                    # read the keys
                    if keys is None:
                        keys = content
                    else:
                        assert keys == content
                    dimSamples = len(keys)-1 + np.sum(params['methods'])
                    samples = np.ones((lengthSamples, dimSamples))
                else:
                    content = content[:-1] + re.split("[\ ]", content[-1][1:-1])
                    content = list(map(lambda x: float(x), content))
                    # read one line value
                    samples[row-1] = gen_sample(content, params)
        all_samples = (samples if all_samples is None else
                       np.concatenate((all_samples, samples), axis=0) )
    
    featuresIndex = list(range(2,2+23)) + list(range(33,33+np.sum(params['methods'])))
    labelsIndex = [keys.index(params["target"])]
    features = all_samples[:, featuresIndex]
    labels = all_samples[:, labelsIndex]
    validIndex = np.where(labels != -1)[0]
    return features[validIndex, :], labels[validIndex]


def listDir(dirName):
    fileList = sorted(os.listdir(dirName))
    fileList = [os.path.join(dirName, fileName) for fileName in fileList if fileName.endswith(".csv")]
    return fileList

def numpy_to_dmatrix(features, labels, params):
    """
    transfer numpy to xgboost DMatrix format
    """
    ratio = 0.8
    numberSamples = features.shape[0]
    featuresTrain = features[:int(numberSamples*ratio),:]
    lablesTrain = labels[:int(numberSamples*ratio),:]
    featuresTest = features[int(numberSamples*ratio):,:]
    lablesTest = labels[int(numberSamples*ratio):,:]
    
    targetName = params['target']
    with open(f'./data/xgboost_{targetName}.txt.train', 'w') as f:
        for feature, label in zip(featuresTrain, lablesTrain):
            f.write(f"{label[0]}")
            for indice, value in enumerate(feature):
                f.write(f" {indice}:{value}")
            f.write("\n")
    
    with open(f'./data/xgboost_{targetName}.txt.test', 'w') as f:
        for feature, label in zip(featuresTest, lablesTest):
            f.write(f"{label[0]}")
            for indice, value in enumerate(feature):
                f.write(f" {indice}:{value}")
            f.write("\n")
        

if __name__ == '__main__':
    dirName = './data/no_pre3/csv'
    params = {'target':'TI'}
    # acc,B_acc,SPD,DIC,EOD,AOD,TI,UF
    features, labels = load_csv(listDir(dirName), params)
    print(features.shape, labels.shape)
    numpy_to_dmatrix(features, labels, params)