import xgboost as xgb
import pickle
from compareModel_utils import dataFeat2CombMatrix, targetMap
from data_utils import onehot_to_class
import random

def generateGT(features, keys):
    """
    args:
        features: 80 * dimFeat
        keys: keys for features
    return:
        dataFeat: same data feature for the features
        fullRankings: List of tuples, tuple: ([2,1,3], values of acc and fairness, keys for values)
    """
    dataFeat = features[0, 13:13+23]
    targetKeys = list(targetMap.keys())
    targetIndexs = []
    for targetKey in targetKeys:
        targetIndexs.append(keys.index(targetMap[targetKey]))
    
    fullRankings = []
    for feature in features:
        onehotList = feature[0:0+13]
        classList = onehot_to_class(onehotList)
        fullRankings.append( (tuple(classList) , feature[targetIndexs], targetKeys) )
    
    fullRankings.sort()
    return dataFeat, fullRankings

def getFullRankingGT(params) -> list:
    """
    args:
        params: 
            recordsFile: original records file path
            saveto: file to save full ranking gt
            methods: [5,4,4]
            fileName: 'vectors_compas_sex_.txt'
    """
    recordsFile = params['recordsFile']
    with open(recordsFile, 'rb') as f:
        records = pickle.load(f)
    features, keys = records[params['fileName']]
    
    fullRankingSamples = []
    preFeat = None
    numRecord = 0
    count = 0
    for index, feature in enumerate(features):
        datasetFeat = tuple(feature[13:13+23])
        if preFeat != datasetFeat:
            preFeat = datasetFeat
            if count == 80:
                # catch one full ranking
                numRecord += 1
                fullRankingOne = generateGT(features[index-80:index, :], keys)
                fullRankingSamples.append(fullRankingOne)
            count = 1
        else:
            count += 1
    print(f"{numRecord} samples generated")
    return fullRankingSamples

def predictComb(params, dataFeats):
    """
    args:
        params: 
            modelLoadFrom:
            saveto: 
        dataFeats: n x 23 
    """
    
    # construct xgboost format data for all 80x80 combination
    dataFeat2CombMatrix(dataFeats, params)
    
    # load xgboost model and predict
    with open(params['modelLoadFrom'], 'rb') as f:
        bst = pickle.load(f)
    xgbData = xgb.DMatrix(params['saveto'])
    predict = bst.predict(xgbData)
    predict = predict.reshape( (80, 80) )
    return predict


if __name__ == '__main__':

    params = {}
    params['recordsFile'] = "/mnt/svm/code/Fairness/Haipei_80/records_seperate.pkl"
    params['methods'] = [5, 4, 4]
    params['saveto'] = './data/fullRankingGT.pkl'
    params['fileName'] = 'vectors_compas_sex_.txt'
    fullRankingSamples = getFullRankingGT(params)

    fullRankingPredicts = []
    for sample in fullRankingSamples:
        onePredict = {}
        params = {}
        for measure in targetMap.keys():
            params['modelLoadFrom'] = f'./models/compas_race_{measure}'
            params['saveto'] = './data/combination.txt'
            prediction = predictComb(params, sample[0].reshape((1,-1)))
            onePredict[measure] = prediction
        
        fullRankingPredicts.append(onePredict)
    
    saveto = "./data/fullRanking_compas_race_to_sex.pkl"
    with open(saveto, 'wb') as f:
        pickle.dump( (fullRankingSamples, fullRankingPredicts), f )
    """
    params = {}
    params['recordsFile'] = "/mnt/svm/code/Fairness/Haipei_80/records_seperate.pkl"
    params['methods'] = [5, 4, 4]
    params['saveto'] = './data/fullRankingGT.pkl'
    params['fileName'] = 'vectors_compas_race_.txt'
    fullRankingSamples = getFullRankingGT(params)
    
    fullRankingPredicts = []
    for sample in fullRankingSamples:
        onePredict = {}
        params = {}
        for measure in targetMap.keys():
            params['modelLoadFrom'] = f'./models/compas_sex_{measure}'
            params['saveto'] = './data/combination.txt'
            prediction = predictComb(params, sample[0].reshape((1,-1)))
            onePredict[measure] = prediction
        
        fullRankingPredicts.append(onePredict)
    
    saveto = "./data/fullRanking_compas_sex_to_race.pkl"
    with open(saveto, 'wb') as f:
        pickle.dump( (fullRankingSamples, fullRankingPredicts ), f )
    """