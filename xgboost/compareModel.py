import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from compareModel_utils import loadParial, prepareTxt, txt2dict, dict2pairs
import matplotlib.pyplot as plt
import pickle

# the reference parameters for traning a random forest
# see https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
paramsTree = {
  'colsample_bynode': 0.8,
  'learning_rate': 0.3,
  'max_depth': 10,
  'num_parallel_tree': 3000,
  'objective': 'binary:logistic',
  'subsample': 0.7,
  'tree_method': 'gpu_hist',
  'verbosity': 1
}

def classifier(params, paramsTree):
    fileRoot = params['saveto']
    modelName = params['modelName']
    modelsaveto = os.path.join('./', 'models' ,modelName)
    trainFile = os.path.join(fileRoot, 'xgboostTrain.txt')
    testFile = os.path.join(fileRoot, 'xgboostTest.txt')
    dtrain = xgb.DMatrix(trainFile)
    dtest = xgb.DMatrix(testFile)

    bst = xgb.train(paramsTree, dtrain, num_boost_round=1)
    with open(modelsaveto, 'wb') as f:
        pickle.dump(bst, f)

    pred_train = np.array(bst.predict(dtrain))
    gt_train = np.array(dtrain.get_label())
    pred_train = pred_train > 0.5
    print("train acc :", float(sum(gt_train == pred_train)) / len(pred_train) )


    pred = bst.predict(dtest)
    gt = np.array(dtest.get_label())
    pred = np.array(pred)
    pred = pred > 0.5
    
    print("test acc :", float(sum(gt == pred)) / len(pred) )

"""
def classifierTrees(params):
    targetName = params['target']
    trainattr = params['trainattr']
    testattr = params['testattr']
    dtrain = xgb.DMatrix(f'/mnt/svm/code/Fairness/Haipei_80/xgboostFormat/splitted/vectors_compas_{trainattr}_{targetName}.txt')
    dtest = xgb.DMatrix(f'/mnt/svm/code/Fairness/Haipei_80/xgboostFormat/splitted/vectors_compas_{testattr}_{targetName}.txt')
    
    bst = xgb.train(paramsTree, dtrain, num_boost_round=1)
    pred = bst.predict(dtest)
    gt = np.array(dtest.get_label())
    pred = np.array(pred)
    pred = pred > 0.5
    
    print("test acc :", float(sum(gt == pred)) / len(pred) )
"""

def constructTrainFile(fileBase, start, end):
    fileName = f"/mnt/svm/code/Fairness/Haipei_80/xgboostFormat/{fileBase}.txt"
    sampledFile = f"/mnt/svm/code/Fairness/Haipei_80/xgboostFormat/splitted/{fileBase}.txt"
    writeF = open(sampledFile, 'w')
    with open(fileName, 'r') as f:
        for i in tqdm.tqdm(range(end)):
            line = f.readline()
            if i >= start:
                writeF.write(line)
            
    writeF.close()
    return None

if __name__ == "__main__":
    # files : ['vectors_adult_race_.txt', 'vectors_adult_sex_.txt', 'vectors_compas_race_.txt', 'vectors_compas_sex_.txt']
    labels = ["B_acc", "acc", "SPD", "DIC", "EOD", "AOD", "TI", "UF"]
    params = {'saveto': "/mnt/svm/code/Fairness/Haipei_80", 'sampleN': 100000}
    #records_seperate = txt2dict()
    #dict2pairs(records_seperate, params)
    for target in labels:
        # prepare data for xgboost model
        print("target", target)
        params = {'saveto': "/mnt/svm/code/Fairness/Haipei_80", 'sampleN': 100000}
        dictData = loadParial(params)
        params = {  'target':target, 
                    'mode': "mixup", 
                    'saveto': "/mnt/svm/code/Fairness/Haipei_80/xgboostFormat",
                    'trainRatio': 0.7,
                    'featSelect': list(range(37)) + list(range(47,49))
                }
        prepareTxt(dictData, params)
        
        # xgboost model params
        paramsTree = {
            'colsample_bynode': 0.8,
            'learning_rate': 0.1,
            'max_depth': 10,
            'num_parallel_tree': 2000,
            'objective': 'binary:logistic',
            'subsample': 0.75,
            'tree_method': 'gpu_hist',
            'verbosity': 0
        }
        print('xgb params:')
        print(paramsTree)
        params = {'saveto': "/mnt/svm/code/Fairness/Haipei_80/xgboostFormat",
                    'modelName': f"mixup_{target}"}
        classifier(params, paramsTree)