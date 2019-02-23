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
#============================
outputPath = "result.csv"
#============================

print("[+]Begin....")
trainData = get_data("data/processed/trainData.csv")
print("[+]Load TrainData Successed!")
# trainLabel = get_data("data/processed/trainLabel.csv")
# print("[+]Load TrainLabel Successed!")
testData = get_data("data/processed/testData.csv")
print("[+]Load TestData Successed!")
# print(testData.info())
# print(trainData.info())
# print(trainLabel.info())
#===================================
current_service = trainData["current_service"].unique()
label_to_current_service = dict(zip(range(0, len(current_service)), current_service))
current_service_to_label = dict(zip(current_service, range(0, len(current_service))))

trainData = trainData.replace({"current_service":current_service_to_label})
trainLabel = trainData["current_service"].copy()
trainLabel = pd.DataFrame(trainLabel)

trainData = trainData.drop(["current_service"], axis=1)

testDataUser = testData["user_id"].copy()
testData = testData.drop(["user_id"],axis=1)
#trainLabel = np.array(trainLabel).squeeze()
trainLabel = trainLabel.copy()
# print(type(trainLabel))
# print(trainLabel)
# exit()
#===================================






lgbParams = {
    'learning_rate': 0.3,
    "lambda_l2": 0.25,
    # "max_depth": 7,
    'num_leaves': 128,
    # 'bagging_fraction':0.8,
    # 'feature_fraction':0.6,
    "objective": "multiclass",
    "num_class": 11,
    'seed': 42,
}

xgbParams={
    'objective': 'multi:softmax',
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 1,
    'alpha':1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel':0.7,
    #'min_child_weight': 3,
    # 'silent': 1,
    'eta': 0.3,
    'seed': 42,
    'num_class':11,
    'eval_metric':'mlogloss',
}

# xx_score = []
# cv_pred = []
# splitsNum = 5
# randomSeed = 42
# numBoostRound = 10#50
# earlyStoppingRounds = 5

# skf = StratifiedKFold(n_splits=splitsNum, random_state=randomSeed, shuffle=True)
#
# for index,(trainIndex,testIndex) in enumerate(skf.split(trainData,trainLabel)):
#     print("[+] " + str(index) + " iteration")
#     X_train = trainData.loc[trainIndex]
#     X_valid = trainData.loc[testIndex]
#     Y_train = trainLabel.loc[trainIndex]
#     Y_valid = trainLabel.loc[testIndex]
#
#     train = xgb.DMatrix(X_train,Y_train)
#     test = xgb.DMatrix(X_valid,Y_valid)
#     watchList = [(test,"eval")]
#     model = xgb.train(xgbParams,train,num_boost_round=numBoostRound, evals=watchList,
#                     early_stopping_rounds=earlyStoppingRounds)#, verbose_eval=1)#,feval=f1_score_vali)
#     testPredict = model.predict(xgb.DMatrix(X_valid))
#     print(np.unique(testPredict))
#     #testPredict = [np.argmax(x) for x in testPredict]
#     xx_score.append(f1_score(Y_valid, testPredict, average='macro'))
#
#     testDataPredict = model.predict(xgb.DMatrix(testData))
#     #testDataPredict = [np.argmax(x) for x in testDataPredict]
#
#     if index == 0:
#         cv_pred = np.array(testDataPredict).reshape(-1, 1)
#     else:
#         cv_pred = np.hstack((cv_pred, np.array(testDataPredict).reshape(-1, 1)))

trainDataAll = trainData.copy()
trainData = pd.DataFrame()
testDataAll = testData.copy()
testData = pd.DataFrame()
featureGain = {}
featureScore = {}
lastScore = 0
count = 0
for feature in trainDataAll.columns:
    #if count == 10:break
    print(feature)
    trainData = pd.concat([trainData,trainDataAll[feature]],axis=1)
    testData = pd.concat([testData, testDataAll[feature]], axis=1)
    print(trainData.shape)
    print(testData.shape)

    xx_score = []
    cv_pred = []
    splitsNum = 5
    randomSeed = 42
    numBoostRound = 200  # 50
    earlyStoppingRounds = 60

    #=====================================================================================
    skf = StratifiedKFold(n_splits=splitsNum, random_state=randomSeed, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(trainData, trainLabel)):
        print("[+] "+str(index)+" iteration")
        #print('第' + str(index) + '次遍历：')
        X_train, X_valid, Y_train, Y_valid = trainData.loc[train_index], trainData.loc[test_index], trainLabel.loc[
            train_index], trainLabel.loc[test_index]
        train_data = lgb.Dataset(X_train, label=Y_train)
        validation_data = lgb.Dataset(X_valid, label=Y_valid)
        clf = lgb.train(lgbParams, train_data, num_boost_round=numBoostRound, valid_sets=[validation_data],
                        early_stopping_rounds=earlyStoppingRounds, feval=f1_score_vali, verbose_eval=1)
        # clf.save_model("modelllll.txt")
        xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
        xx_pred = [np.argmax(x) for x in xx_pred]
        xx_score.append(f1_score(Y_valid, xx_pred, average='macro'))

        y_test = clf.predict(testData, num_iteration=clf.best_iteration)
        y_test = [np.argmax(x) for x in y_test]
        if index == 0:
            cv_pred = np.array(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

    # 投票
    cv_pred = cv_pred.astype(int)
    submit = []
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))
    df_test = pd.DataFrame()
    df_test['id'] = list(testDataUser.unique())
    df_test['predict'] = submit
    df_test['predict'] = df_test['predict'].map(label_to_current_service)
    df_test.to_csv(outputPath, index=False)
    meanScore = np.mean(xx_score)
    result = str(xx_score) + " "+ str(np.mean(xx_score))
    print(xx_score, meanScore)
    featureScore[str(feature)+str(count)] = meanScore
    featureGain[str(feature)+str(count)] = meanScore - lastScore
    lastScore = meanScore
    count+=1
    #===============================================================================

print("[+]featureGain:"+str(featureGain))
print("[+]featureScore:"+str(featureScore))

subject = "[DianXin]"
message = str(featureGain) + str(featureScore)
Send_Email(subject, message, ["609543803@qq.com"])
