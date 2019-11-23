import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


# the reference parameters for traning a random forest
# see https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
paramsTree = {
  'colsample_bynode': 0.8,
  'learning_rate': 1,
  'max_depth': 10,
  'num_parallel_tree': 1000,
  'objective': 'reg:squarederror',
  'subsample': 0.9,
  'tree_method': 'gpu_hist',
  'verbosity': 1
}

def regressTrees(params):
    targetName = params['target']

    dtrain = xgb.DMatrix(f'./data/xgboost_{targetName}.txt.train')
    dtest = xgb.DMatrix(f'./data/xgboost_{targetName}.txt.test')

    bst = xgb.train(paramsTree, dtrain, num_boost_round=1)
    pred = bst.predict(dtest)


    gt = np.array(dtest.get_label())
    mean = np.mean(gt)
    meanPred = np.ones_like(gt)*mean

    pred = np.array(pred)
    print("Ground True: ", gt[:5])
    print("Predictions: ", pred[:5])

    l2loss = np.power(np.sum(np.power(gt-pred, 2)), 0.5)/gt.shape[0]
    l2lossMean = np.power(np.sum(np.power(gt-meanPred, 2)), 0.5)/gt.shape[0]
    print(f"l2 loss of pred is {l2loss}")
    print(f"l2 loss of mean is {l2lossMean}")


if __name__ == "__main__":
    params = {}
    for target in ['acc', 'AOD', 'B_acc', 'DIC', 'EOD', 'SPD', 'TI', 'UF']:
        print("#"*10, target, '#'*10)
        params['target'] = target
        regressTrees(params)