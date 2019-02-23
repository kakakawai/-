import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from mail import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from Layers import *

# ============================
layerName = "xgb_3_1"
outputPath = "result_" + str(layerName) + ".csv"
# ============================
xgbParams = {
    'objective': 'multi:softprob',
    'learning_rate':0.05,
    'gamma': 0.1,
    'max_depth': 12,
    'lambda': 1,
    'alpha': 1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.9,
    'seed': 58,
    'num_class': 11,
    'eval_metric': 'mlogloss',
    'base_score':0.5
}

cvParams = {
    "splitsNum":10,
    "randomSeed": 144,
    "numBoostRound":3000,
    "earlyStoppingRounds":2000,
}
# ============================
if __name__ == "__main__":
    print("[+]Begin....")

    trainData = get_data("data/processed/trainData.csv")
    print("[+]Load TrainData Successed!")

    testData = get_data("data/processed/testData.csv")
    print("[+]Load TestData Successed!")

    trainAddsData = get_data("data/processed/trainData_lgb_2_1.csv")
    print("[+]Load trainData_xgb1.csv Successed!")

    testAddsData = get_data("data/processed/testData_lgb_2_1.csv")
    print("[+]Load testData_xgb1.csv Successed!")

    # ===================================
    trainData, trainLabel, testData, testDataUser, label_to_current_service = get_layer_data(trainData, testData)
    cv_trainProb, cv_testProb, cv_pred, xx_score = XGBTrain(trainData, trainLabel, testData, xgbParams, cvParams)

    save_adds_data(cv_trainProb, cv_testProb, cvParams["splitsNum"], layerName=layerName)

    layer_predict(cv_pred, testDataUser, label_to_current_service, outputPath=outputPath, layerName=layerName)

    result_output(xx_score, layerName=layerName)
