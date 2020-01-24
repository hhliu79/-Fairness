
import pickle
import xgboost as xgb
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt
import pickle


def featureInfo():
    """
    ‘weight’: the number of times a feature is used to split the data across all trees.
    ‘gain’: the average gain across all splits the feature is used in.
    ‘cover’: the average coverage across all splits the feature is used in.
    ‘total_gain’: the total gain across all splits the feature is used in.
    ‘total_cover’: the total coverage across all splits the feature is used in.
    """
    for itype in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
        for modelTarget in ["acc", 'AOD', "B_acc", "DIC", "EOD", "SPD", "TI", "UF"]:
            with open(f"./models/{modelTarget}", 'rb') as f:
                model = pickle.load(f)
            featScores = model.get_score(importance_type=itype)
            featScores = [(int(key[1:]), value) for key, value in featScores.items()]
            featScores.sort()
            print(featScores)
            featScores = [feat[1] for feat in featScores]
            plt.bar( np.arange(len(featScores)), featScores)
            # plt.show()
            plt.savefig(f'./figs/{modelTarget}_{itype}.png')
            plt.close()
            
if __name__ == '__main__':
    featureInfo()