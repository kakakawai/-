# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from mail import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def get_data(path):
    Data = pd.read_csv(path)
    return Data

def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali ** 2, True

def get_layer_data(trainData,testData):
    current_service = trainData["current_service"].unique()
    label_to_current_service = dict(zip(range(0, len(current_service)), current_service))
    current_service_to_label = dict(zip(current_service, range(0, len(current_service))))

    trainData = trainData.replace({"current_service":current_service_to_label})
    trainLabel = trainData["current_service"].copy()
    print("[+]train Label shape:" + str(np.shape(trainLabel)))

    trainData = trainData.drop(["current_service"], axis=1)
    testDataUser = testData["user_id"].copy()
    testData = testData.drop(["user_id"],axis=1)

    return trainData,trainLabel,testData,testDataUser,label_to_current_service

def combine_layer_data(trainData,testData,trainAddsData,testAddsData):

    trainData = pd.concat([trainData,trainAddsData],axis=1)
    print("[+]combine train data shape:"+str(trainData.shape))


    testData = pd.concat([testData,testAddsData],axis=1)
    print("[+]combine test data shape:"+str(testData.shape))

    return trainData,testData


def LGBTrain(trainData, trainLabel,testData, lgbParams, cvParams):
    xx_score = []
    cv_trainProb = []
    cv_testProb = []
    cv_pred = []

    splitsNum = cvParams["splitsNum"]
    randomSeed = cvParams["randomSeed"]
    numBoostRound = cvParams["numBoostRound"]
    earlyStoppingRounds = cvParams["earlyStoppingRounds"]

    skf = StratifiedKFold(n_splits=splitsNum, random_state=randomSeed, shuffle=True)
    for index, (trainIndex, testIndex) in enumerate(skf.split(trainData, trainLabel)):
        print("[+] " + str(index) + " iteration")
        X_train = trainData.loc[trainIndex]
        X_valid = trainData.loc[testIndex]
        Y_train = trainLabel.loc[trainIndex]
        Y_valid = trainLabel.loc[testIndex]

        train = lgb.Dataset(X_train, label=Y_train)
        test = lgb.Dataset(X_valid, label=Y_valid)
        model = lgb.train(lgbParams, train, num_boost_round=numBoostRound, valid_sets=[test],
                          early_stopping_rounds=earlyStoppingRounds, feval=f1_score_vali, verbose_eval=1)
        testPredict = model.predict(X_valid, num_iteration=model.best_iteration)
        testPredict = [np.argmax(x) for x in testPredict]
        xx_score.append(f1_score(Y_valid, testPredict, average='macro'))

        trainDataProb = model.predict(trainData, num_iteration=model.best_iteration)
        print(trainDataProb[:5])

        testDataProb = model.predict(testData, num_iteration=model.best_iteration)
        print(testDataProb[:5])
        testDataPredict = [np.argmax(x) for x in testDataProb]

        if index == 0:
            cv_pred = np.array(testDataPredict).reshape(-1, 1)
            cv_trainProb = np.array(trainDataProb)
            cv_testProb = np.array(testDataProb)

        else:
            cv_pred = np.hstack((cv_pred, np.array(testDataPredict).reshape(-1, 1)))
            cv_trainProb = cv_trainProb + np.array(trainDataProb)
            cv_testProb = cv_testProb + np.array(testDataProb)

        print("=======cv_result=========")
        print(cv_pred[:5])
        print(cv_trainProb[:5])
        print(cv_testProb[:5])
    return cv_trainProb, cv_testProb, cv_pred, xx_score

def XGBTrain(trainData, trainLabel,testData, xgbParams, cvParams):
    xx_score = []
    cv_trainProb = []
    cv_testProb = []
    cv_pred = []

    splitsNum = cvParams["splitsNum"]
    randomSeed = cvParams["randomSeed"]
    numBoostRound = cvParams["numBoostRound"]
    earlyStoppingRounds = cvParams["earlyStoppingRounds"]

    skf = StratifiedKFold(n_splits=splitsNum, random_state=randomSeed, shuffle=True)
    for index, (trainIndex, testIndex) in enumerate(skf.split(trainData, trainLabel)):
        print("[+] " + str(index) + " iteration")
        X_train = trainData.loc[trainIndex]
        X_valid = trainData.loc[testIndex]
        Y_train = trainLabel.loc[trainIndex]
        Y_valid = trainLabel.loc[testIndex]

        train = xgb.DMatrix(X_train, Y_train)
        test = xgb.DMatrix(X_valid, Y_valid)
        watchList = [(test, "eval")]
        model = xgb.train(xgbParams, train,num_boost_round=numBoostRound, evals=watchList,early_stopping_rounds=earlyStoppingRounds)
        testPredict = model.predict(xgb.DMatrix(X_valid))
        testPredict = [np.argmax(x) for x in testPredict]
        xx_score.append(f1_score(Y_valid, testPredict, average='macro'))

        trainDataProb = model.predict(xgb.DMatrix(trainData))
        print(trainDataProb[:5])

        testDataProb = model.predict(xgb.DMatrix(testData))
        print(testDataProb[:5])
        testDataPredict = [np.argmax(x) for x in testDataProb]

        if index == 0:
            cv_pred = np.array(testDataPredict).reshape(-1, 1)
            cv_trainProb = np.array(trainDataProb)
            cv_testProb = np.array(testDataProb)

        else:
            cv_pred = np.hstack((cv_pred, np.array(testDataPredict).reshape(-1, 1)))
            cv_trainProb = cv_trainProb + np.array(trainDataProb)
            cv_testProb = cv_testProb + np.array(testDataProb)

        print("=======cv_result=========")
        print(cv_pred[:5])
        print(cv_trainProb[:5])
        print(cv_testProb[:5])
    return cv_trainProb, cv_testProb, cv_pred, xx_score

def XGBTrainSK(trainData, trainLabel,testData, xgbParams, cvParams):
    xx_score = []
    cv_trainProb = []
    cv_testProb = []
    cv_pred = []

    splitsNum = cvParams["splitsNum"]
    randomSeed = cvParams["randomSeed"]
    numBoostRound = cvParams["numBoostRound"]
    earlyStoppingRounds = cvParams["earlyStoppingRounds"]
    eval_metric = cvParams["eval_metric"]

    objective=xgbParams["objective"]
    learning_rate=xgbParams["learning_rate"]
    gamma=xgbParams["gamma"]
    max_depth=xgbParams["max_depth"]
    lambdaa=xgbParams["lambda"]
    alpha=xgbParams["alpha"]
    subsample=xgbParams["subsample"]
    colsample_bytree=xgbParams["colsample_bytree"]
    colsample_bylevel=xgbParams["colsample_bylevel"]
    seed=xgbParams["seed"]
    num_class=xgbParams["num_class"]
    base_score=xgbParams["base_score"]
    n_estimators = xgbParams["n_estimators"]
    n_jobs = xgbParams["n_jobs"]


    skf = StratifiedKFold(n_splits=splitsNum, random_state=randomSeed, shuffle=True)
    for index, (trainIndex, testIndex) in enumerate(skf.split(trainData, trainLabel)):
        print("[+] " + str(index) + " iteration")
        X_train = trainData.loc[trainIndex]
        X_valid = trainData.loc[testIndex]
        Y_train = trainLabel.loc[trainIndex]
        Y_valid = trainLabel.loc[testIndex]

        # train = xgb.DMatrix(X_train, Y_train)
        # test = xgb.DMatrix(X_valid, Y_valid)
        #watchList = [(test, "eval")]
        watchList = [(X_valid,Y_valid)]
        model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective=objective,
            n_jobs=n_jobs,
            gamma=gamma,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            reg_alpha=alpha,
            reg_lambda=lambdaa,
            base_score=base_score,
            random_state=seed

        )
        model.fit(X_train,Y_train,eval_set=watchList,eval_metric=eval_metric,early_stopping_rounds=earlyStoppingRounds)
        #model = xgb.train(xgbParams, train,num_boost_round=numBoostRound, evals=watchList,early_stopping_rounds=earlyStoppingRounds)
        testPredict = model.predict(X_valid)
        testPredict = [np.argmax(x) for x in testPredict]
        xx_score.append(f1_score(Y_valid, testPredict, average='macro'))

        trainDataProb = model.predict(trainData)
        print(trainDataProb[:5])

        testDataProb = model.predict(testData)
        print(testDataProb[:5])
        testDataPredict = [np.argmax(x) for x in testDataProb]

        if index == 0:
            cv_pred = np.array(testDataPredict).reshape(-1, 1)
            cv_trainProb = np.array(trainDataProb)
            cv_testProb = np.array(testDataProb)

        else:
            cv_pred = np.hstack((cv_pred, np.array(testDataPredict).reshape(-1, 1)))
            cv_trainProb = cv_trainProb + np.array(trainDataProb)
            cv_testProb = cv_testProb + np.array(testDataProb)

        print("=======cv_result=========")
        print(cv_pred[:5])
        print(cv_trainProb[:5])
        print(cv_testProb[:5])
    return cv_trainProb, cv_testProb, cv_pred, xx_score

def save_adds_data(trainProb,testProb,splitsNum,layerName):
    trainProb = trainProb/splitsNum
    trainProb = pd.DataFrame(trainProb)
    trainProb.columns =  [layerName+"_label_"+str(i) for i in range(11)]#["lgb_3_1_label_0","lgb_3_1_label_1","lgb_3_1_label_2","lgb_3_1_label_3","lgb_3_1_label_4","lgb_3_1_label_5","lgb_3_1_label_6","lgb_3_1_label_7","lgb_3_1_label_8","lgb_3_1_label_9","lgb_3_1_label_10"]
    print("[+]cv_trainProb shape:"+str(trainProb.shape))
    print(trainProb[:5])
    #trainData = pd.concat([trainData,cv_prob,trainInfo],axis=1)
    trainProb.to_csv("data/processed/trainData_"+layerName+".csv",index=False)
    print("[+]trainData "+layerName+" saved!")

    testProb = testProb/splitsNum
    testProb = pd.DataFrame(testProb)
    testProb.columns = [layerName+"_label_"+str(i) for i in range(11)]#["lgb_3_1_label_0","lgb_3_1_label_1","lgb_3_1_label_2","lgb_3_1_label_3","lgb_3_1_label_4","lgb_3_1_label_5","lgb_3_1_label_6","lgb_3_1_label_7","lgb_3_1_label_8","lgb_3_1_label_9","lgb_3_1_label_10"]
    print("[+]testProb shape:"+str(testProb.shape))
    print(testProb[:5])
    testProb.to_csv("data/processed/testData_"+layerName+".csv",index=False)
    print("[+]testData "+layerName+" saved!")

def layer_predict(cv_pred,testDataUser,label_to_current_service,outputPath,layerName):
    # 投票
    cv_pred = cv_pred.astype(int)
    #submit = []
    #for line in cv_pred:
    #   submit.append(np.argmax(np.bincount(line)))
    # df_test = pd.DataFrame()
    # df_test['id'] = list(testDataUser.unique())
    # df_test['predict'] = submit
    # df_test['predict'] = df_test['predict'].map(label_to_current_service)
    # df_test.to_csv(outputPath, index=False)
    cv_pred = pd.DataFrame(cv_pred)
    length = cv_pred.shape[1]
    cv_pred.columns = [layerName+"_cv_"+str(i) for i in range(length)]
    for i in cv_pred.columns:
        cv_pred[i] = cv_pred[i].map(label_to_current_service)
    cv_pred.to_csv(outputPath, index=False)

def result_output(xx_score,layerName):
    meanScore = np.mean(xx_score)
    result = str(xx_score) + " "+ str(np.mean(xx_score))
    print(xx_score, meanScore)

    subject = "[DianXin]"+layerName
    message = "[+]"+layerName+" testData Predict:" + result
    Send_Email(subject, message, ["609543803@qq.com"])