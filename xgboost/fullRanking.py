import xgboost as xgb
import pickle
from compareModel_utils import dataFeat2CombMatrix

def getFullRankingGT(params):
    """
    args:
        params: 
            recordsFile: original records file path
            saveto: file to save full ranking gt
            methods: [5,4,4]
    """
    recordsFile = params['recordsFile']
    with open(recordsFile, 'rb') as f:
        records = pickle.load(f)
    features, keys = records['vectors_compas_race_.txt']
    
    fullranking = 0
    preFeat = None
    count = 0
    for feature in features:
        datasetFeat = tuple(feature[13:13+23])
        if preFeat != datasetFeat:
            preFeat = datasetFeat
            if count == 80:
                fullranking += 1
            count = 1
        else:
            count += 1
    print(fullranking)

def predictComb(params, dataFeats):
    """
    args:
        params: 
            modelLoadFrom:
        dataFeats: n x 23 
    """
    params = {}
    params['saveto'] = './data/combination.txt'
    dataFeat2CombMatrix(dataFeats, params)
    with open(params['modelLoadFrom'], 'rb') as f:
        bst = pickle.load(f)

    xgbData = xgb.DMatrix(params['saveto'])
    predict = bst.predict(xgbData)
    predict = predict.reshape( (80, 80) )

if __name__ == '__main__':
    params = {}
    params['recordsFile'] = "/mnt/svm/code/Fairness/Haipei_80/records_seperate.pkl"
    params['methods'] = [5, 4, 4]
    params['saveto'] = './data/fullRankingGT.pkl'
    getFullRankingGT(params)