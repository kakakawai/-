# -*- coding:utf-8 -*-
#Author:Mirror
#CreateDate:18.9.6
#ModifiedDate:18.9.6

import pandas as pd
import numpy as np
from preprocess import *
from mail import *
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


#================================
meanFeatures = ["online_time","month_traffic","contract_time","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","former_complaint_num","former_complaint_fee"]
modeFeatures = ["1_total_fee","2_total_fee","3_total_fee","4_total_fee"]
#================================

#def get_data(path,columns):
#     Data = pd.read_csv(path)
#     Data.columns = columns
#     return Data

def get_data(path):
    Data = pd.read_csv(path)
    return Data

def get_service_features(groupData):
    #meanData = groupData[["online_time","1_total_fee","2_total_fee","3_total_fee","4_total_fee","month_traffic","contract_time","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","former_complaint_num","former_complaint_fee"]].mean(axis=1)
    meanData = groupData[meanFeatures].mean(axis=0)
    modeData = groupData[modeFeatures].mode(axis=0).T
    resultData = pd.concat([meanData,modeData])
    #print(resultData.T)
    return resultData.T

def services_features(allTrainData):
    meanData = allTrainData.groupby("current_service").apply(get_service_features)
    meanData = meanData.reset_index().drop(["current_service","level_1"],axis=1)
    #print(meanData.shape)
    return meanData

def featureDiff(Data,featureData):
    print(featureData.columns)
    print(featureData.shape)
    for feature in featureData.columns:
        for i,value in zip(range(featureData.shape[0]),featureData[feature]):
            Data[feature+"Diff"+str(i)] = Data[feature] - value
    return Data

def predictDiscrete(Data,column):
    print("[+]" + column + " predict begin")

    trainData = Data[Data[column] > 0]
    testData = Data[Data[column] == 0]

    trainLabel,trainData = trainData[column].copy(),trainData.drop([column],axis=1)
    testData = testData.drop([column], axis=1)

    model = GradientBoostingClassifier()
    model.fit(trainData,trainLabel)
    gbmPredict = np.array(model.predict(testData),dtype=int).reshape(-1, 1)

    model = RandomForestClassifier()
    model.fit(trainData,trainLabel)
    RFPredict = np.array(model.predict(testData),dtype=int).reshape(-1, 1)

    model = AdaBoostClassifier()
    model.fit(trainData,trainLabel)
    AdaPredict = np.array(model.predict(testData),dtype=int).reshape(-1, 1)

    #Combine
    predictCombine = np.concatenate((gbmPredict,RFPredict,AdaPredict),axis=1)

    Predict = []
    for line in predictCombine:
        Predict.append(np.argmax(np.bincount(line)))
    np.array(Predict).reshape(-1, 1)

    Data[column][Data[column] == 0] = Predict
    Data[column] = Data[column].astype(int)

    return Data

def predictContinue(Data,column):
    print("[+]" + column + " predict begin")

    trainData = Data[Data[column] > 0]
    testData = Data[Data[column] == 0]

    trainLabel,trainData = trainData[column].copy(),trainData.drop([column],axis=1)
    testData = testData.drop([column],axis=1)

    model = GradientBoostingRegressor()
    model.fit(trainData,trainLabel)
    gbmPredict = np.array(model.predict(testData),dtype=float).reshape(-1, 1)

    model = RandomForestRegressor()
    model.fit(trainData,trainLabel)
    RFPredict = np.array(model.predict(testData),dtype=float).reshape(-1, 1)

    model = AdaBoostRegressor()
    model.fit(trainData,trainLabel)
    AdaPredict = np.array(model.predict(testData),dtype=float).reshape(-1, 1)

    #Combine
    PredictCombine = np.concatenate((gbmPredict,RFPredict,AdaPredict),axis=1)
    Predict = np.mean(PredictCombine,axis=1)

    Data[column][Data[column] == 0] = Predict
    Data[column] = Data[column].astype(float)

    return Data

if __name__ == "__main__":

    outputPath = "result.csv"

    print("[+]begin")
    # train_columns = ["service_type","is_mix_service","online_time","1_total_fee","2_total_fee","3_total_fee","4_total_fee","month_traffic","many_over_bill","contract_type","contract_time","is_promise_low_consume","net_service","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","gender","age","complaint_level","former_complaint_num","former_complaint_fee","current_service","user_id"]
    # test_columns = ["service_type", "is_mix_service", "online_time", "1_total_fee", "2_total_fee", "3_total_fee",
    #                 "4_total_fee", "month_traffic", "many_over_bill", "contract_type", "contract_time",
    #                 "is_promise_low_consume", "net_service", "pay_times", "pay_num", "last_month_traffic",
    #                 "local_trafffic_month", "local_caller_time", "service1_caller_time", "service2_caller_time",
    #                 "gender", "age", "complaint_level", "former_complaint_num", "former_complaint_fee", "user_id"]

    trainData = get_data("data/new/train.csv")
    testData = get_data("data/new/test.csv")

    testData["service_type"] = testData["service_type"].replace(3,4)

    current_service = trainData["current_service"].unique()
    #
    label_to_current_service = dict(zip(range(0,len(current_service)),current_service))
    current_service_to_label = dict(zip(current_service,range(0,len(current_service))))

    trainInfo = trainData["current_service"].copy()
    trainInfo = pd.DataFrame(trainInfo)

    trainData = trainData.replace({"current_service":current_service_to_label})
    trainLabel = trainData["current_service"].copy()
    trainLabel = pd.DataFrame(trainLabel)

    trainData = trainData.drop(["current_service","user_id"],axis=1)

    testDataUser = testData["user_id"].copy()
    testData = testData.drop(["user_id"],axis=1)

    spliteIndex = trainData.shape[0]
    combineData = pd.concat([trainData,testData])
    combineData = data_dropNan(combineData)
    combineData = predictContinue(combineData, "age")
    combineData = predictDiscrete(combineData, "gender")
    combineData = data_preprocess(combineData)
    trainData = combineData[:spliteIndex]
    testData = combineData[spliteIndex:]
    # trainData = data_dropNan(trainData)
    # testData = data_dropNan(testData)

    # trainData,testData = predictContinue(trainData,testData,"1_total_fee")
    # trainData, testData = predictContinue(trainData, testData, "2_total_fee")
    # trainData, testData = predictContinue(trainData, testData, "3_total_fee")
    # trainData, testData = predictContinue(trainData, testData, "4_total_fee")


    # trainData, testData = predictContinue(trainData, testData, "age")
    # trainData["age"] = trainData["age"].astype(int)
    # testData["age"] = testData["age"].astype(int)
    # print(trainData["age"])
    # print(testData["age"])
    # trainData, testData = predictDiscrete(trainData, testData, "gender")
    # trainData["gender"] = trainData["gender"].astype(int)
    # testData["gender"] = testData["gender"].astype(int)
    # print(trainData["gender"])
    # print(testData["gender"])


    # trainData = data_preprocess(trainData)
    # #print(trainData.info())
    # testData = data_preprocess(testData)


    allTrainData = pd.concat([trainData,trainLabel],axis=1)
    featureData = services_features(allTrainData)
    # print(type(featureData))
    # featureData.to_csv("data/processed/featureData.csv",index=False)
    # exit()
    trainData = featureDiff(trainData,featureData)
    testData = featureDiff(testData,featureData)

    print(testData.info())
    print(trainData.info())
    print(trainLabel.info())

#========================================================
    trainData = pd.concat([trainData,trainInfo],axis=1)
    testData = pd.concat([testData,testDataUser],axis=1)
    print("[+]trainData shape:"+str(trainData.shape))
    print("[+]testData shape:" + str(testData.shape))
    trainData.to_csv("data/processed/trainData.csv",index=False)
    #trainLabel.to_csv("data/processed/trainLabel.csv",index=False)
    testData.to_csv("data/processed/testData.csv",index=False)
    #testDataUser.to_csv("data/processed/testDataUser.csv",index=False)

    print("[+]preprocess Done!")
    print("[+]model fitting begin!")
    exit()
    #print(testData.shape[0])
#===================================================

    # trainGender = trainData[trainData["gender"]>0]
    # trainTestGender = trainData[trainData["gender"]==0]
    # testTestGender = testData[testData["gender"]==0]
    # trainPredict,testPredict = predictGender(trainGender, trainTestGender,testTestGender)
    # trainData["gender"][trainData["gender"]==0] = trainPredict
    # trainData["gender"] = trainData["gender"].astype(int)
    # testData["gender"][testData["gender"] == 0] = testPredict
    # testData["gender"] = testData["gender"].astype(int)

    #print(trainData["gender"][trainData["gender"] == 0])
    #print(testData["gender"][testData["gender"] == 0])

    # trainGender = trainData[trainData["age"] > 0]
    # trainTestGender = trainData[trainData["age"] == 0]
    # testTestGender = testData[testData["age"] == 0]
    # trainPredict, testPredict = predictAge(trainGender, trainTestGender, testTestGender)
    # trainData["age"][trainData["age"] == 0] = trainPredict
    # trainData["age"] = trainData["age"].astype(int)
    # testData["age"][testData["age"] == 0] = testPredict
    # testData["age"] = testData["age"].astype(int)

    #print(trainData["age"][trainData["age"] == 0])
    #print(testData["age"][testData["age"] == 0])



#===================================================

    def f1_score_vali(preds, data_vali):
        labels = data_vali.get_label()
        preds = np.argmax(preds.reshape(15, -1), axis=0)
        score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
        return 'f1_score', score_vali ** 2, True


    xx_score = []
    cv_pred = []
    params = {
        'learning_rate': 0.1,
        "lambda_l2": 0.25,
        # "max_depth": 7,
        'num_leaves':128,
        # 'bagging_fraction':0.8,
        # 'feature_fraction':0.6,
        "objective": "multiclass",
        "num_class": 15,
        'seed':42,
    }
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(trainData, trainLabel)):
        print('第' + str(index) + '次遍历：')
        X_train, X_valid, Y_train, Y_valid = trainData.loc[train_index], trainData.loc[test_index], trainLabel.loc[train_index], trainLabel.loc[test_index]
        train_data = lgb.Dataset(X_train, label=Y_train)
        validation_data = lgb.Dataset(X_valid, label=Y_valid)
        clf = lgb.train(params, train_data, num_boost_round=3000, valid_sets=[validation_data],
                        early_stopping_rounds=500, feval=f1_score_vali, verbose_eval=1)
        #clf.save_model("modelllll.txt")
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
    submit = []
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))
    df_test = pd.DataFrame()
    df_test['id'] = list(testDataUser.unique())
    df_test['predict'] = submit
    df_test['predict'] = df_test['predict'].map(label_to_current_service)

    df_test.to_csv(outputPath, index=False)
    result = str(xx_score) + str(np.mean(xx_score))
    print(xx_score, np.mean(xx_score))
    subject = "[DianXin]"
    message = result
    Send_Email(subject, message, ["609543803@qq.com"])

#===================================================
    '''
    kf = KFold(n_splits=10)
    kfCount = 0
    Acclist = []

    

    model = lgb.LGBMClassifier()#xgb.XGBClassifier(n_jobs=2)##GradientBoostingClassifier()#MultinomialNB()#tree.DecisionTreeClassifier()
    #LR = LogisticRegression() 0.42
    #svc = svm.SVC()
    #knn = KNeighborsClassifier(n_neighbors=15) 0.74
    #dt = tree.DecisionTreeClassifier() 0.89
    #gnb = GaussianNB() 0.148
    ## mnb = MultinomialNB()
    #ada = AdaBoostClassifier() 0.67
    #gra = GradientBoostingClassifier() #0.90
    #rf = RandomForestClassifier() 0.91
    #lg = lgb.LGBMClassifier() 0.919
    #xgb = xgb.XGBClassifier(n_jobs=2)0.89

    for trainIndex,testIndex in kf.split(trainData):
        kfCount +=1
        kfTrainData = trainData.iloc[trainIndex]
        kfTrainLabel = trainLabel.iloc[trainIndex]
        kfTestData = trainData.iloc[testIndex]
        kfTestLabel = trainLabel.iloc[testIndex]

        model.fit(kfTrainData,kfTrainLabel)
        predict = model.predict(kfTestData)
        predict = predict.tolist()
        kfTestLabel = kfTestLabel.tolist()

        count = 0
        all = len(predict)
        for i in range(all):
            if predict[i] == kfTestLabel[i]:
                count += 1
        Acc = float(count) / all
        Acclist.append(Acc)
        print("[+]Acc on %d fold:%f" % (kfCount, Acc))
    print("[+]Acc mean: %f" % (sum(Acclist) / len(Acclist)))

    model.fit(trainData,trainLabel)
    predict = model.predict(testData)

    with(open(outputPath, mode="w",newline="")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["user_id", "predict"])
        for i in range(len(predict)):
            # pred = 0 if pred < 0.5 else pred
            writer.writerow([testDataUser[i], label_to_current_service[predict[i]]])
    '''