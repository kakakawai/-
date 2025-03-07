# -*- coding:utf-8 -*-
from Layers import *

#============================
layerName = "lgb_3_2"
outputPath = "result_"+str(layerName)+".csv"
#============================
lgbParams = {
        'learning_rate': 0.05,
        "lambda_l2": 0.25,
        # "max_depth": 7,
        'num_leaves': 128,
        # 'bagging_fraction':0.8,
        # 'feature_fraction':0.6,
        "objective": "multiclass",
        "num_class": 11,
        'seed': 42,
    }
cvParams = {
    "splitsNum":10,  # 5
    "randomSeed": 369,
    "numBoostRound":3000,
    "earlyStoppingRounds":2000,
}
#==============================
#=========================================================================================

if __name__ =="__main__":
    print("[+]Begin....")

    trainData = get_data("data/processed/trainData.csv")
    print("[+]Load TrainData Successed!")

    testData = get_data("data/processed/testData.csv")
    print("[+]Load TestData Successed!")

    # trainAddsData = get_data("data/processed/trainData_lgb_2_1.csv")
    # print("[+]Load trainData_xgb1.csv Successed!")
    #
    # testAddsData = get_data("data/processed/testData_lgb_2_1.csv")
    # print("[+]Load testData_xgb1.csv Successed!")
    #===================================
    trainData, trainLabel, testData, testDataUser, label_to_current_service=get_layer_data(trainData,testData)
    # trainData,testData = combine_layer_data(trainData, testData, trainAddsData, testAddsData)
    cv_trainProb, cv_testProb, cv_pred, xx_score=LGBTrain(trainData, trainLabel, testData, lgbParams, cvParams)

    save_adds_data(cv_trainProb, cv_testProb,cvParams["splitsNum"], layerName = layerName)

    layer_predict(cv_pred, testDataUser, label_to_current_service, outputPath=outputPath,layerName=layerName)

    result_output(xx_score, layerName=layerName)

