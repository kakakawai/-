# -*- coding:utf-8 -*-
#Author:Mirror
#CreateDate:18.9.6
#ModifiedDate:18.9.6

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


def get_data(path,columns):
    Data = pd.read_csv(path)
    Data.columns = columns
    return Data

def get_ageRange(data):
    if data<=10:
        return 0
    elif 10<data<=25:
        return 1
    elif 25<data<=50:
        return 2
    elif 50<data<=70:
        return 3
    elif 70<data<99:
        return 4
    elif data == 99:
        return 5

def get_onlineTimeRange(data):
    if data <20:
        return 0
    elif 20<=data<50:
        return 1
    elif 50<=data<90:
        return 2
    elif 90<= data:
        return 3

def total_fee_replace_mean(data):
    columns = ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]
    for feature in columns:
        features = ["1_total_fee","2_total_fee","3_total_fee","4_total_fee"]
        features.remove(feature)
        if data[feature] ==0:
            #print(data[features][data[features]>0])
            data[feature] = data[features][data[features]>0].mean()
            if np.isnan(data[feature]): data[feature]=0
            data[feature] = round(data[feature],4)
            #print(feature+" "+str(data[feature]))
    return data

def data_dropNan(Data):
    # print(Data)
    # print(trainData[trainData["2_total_fee"]=="\\N"])
    two_total_fee_mean = Data[Data["2_total_fee"].apply(lambda x: x != "\\N")]["2_total_fee"].astype(float).mean()
    Data["2_total_fee"].replace("\\N", two_total_fee_mean, inplace=True)
    Data["2_total_fee"] = Data["2_total_fee"].astype(float)
    # print(trainData["2_total_fee"])

    three_total_fee_mean = Data[Data["3_total_fee"].apply(lambda x: x != "\\N")]["3_total_fee"].astype(float).mean()
    Data["3_total_fee"].replace("\\N", three_total_fee_mean, inplace=True)
    Data["3_total_fee"] = Data["3_total_fee"].astype(float)

    # age_mean = Data[Data["age"].apply(lambda x:x != "\\N")]["age"].astype(float).mean()
    Data["age"].replace("\\N", 0, inplace=True)
    Data["age"] = Data["age"].astype(float)

    Data["gender"].replace("\\N", 0, inplace=True)
    Data["gender"] = Data["gender"].astype(int)

    # Data["ageRange"] = Data["age"].apply(get_ageRange)
    # dummyFeature = pd.get_dummies(Data[["ageRange"]])
    # Data = Data.drop(["ageRange","age"],axis=1)
    # Data = pd.concat([Data,dummyFeature],axis=1)

    # print(Data.info())
    # print(Data[Data["contract_time"] == -1].shape[0])
    dummyFeature = pd.get_dummies(Data[["complaint_level", "contract_type", "service_type", "net_service"]])
    Data = Data.drop(["complaint_level", "contract_type", "service_type", "net_service"], axis=1)
    Data = pd.concat([Data, dummyFeature], axis=1)

    # Data["1_total_fee"].apply(lambda x: max(x, 0))
    # Data["2_total_fee"].apply(lambda x: max(x, 0))
    # Data["3_total_fee"].apply(lambda x: max(x, 0))
    # Data["4_total_fee"].apply(lambda x: max(x, 0))
    Data["1_total_fee"] = Data["1_total_fee"].apply(abs)
    Data["2_total_fee"] = Data["2_total_fee"].apply(abs)
    Data["3_total_fee"] = Data["3_total_fee"].apply(abs)
    Data["4_total_fee"] = Data["4_total_fee"].apply(abs)

    #============new=====================
    Data = Data.apply(total_fee_replace_mean,axis=1)
    print("[+]0replace mean over")

#==============0.9 Quantile Replace (except "age")================
    #columns = ["online_time","1_total_fee","2_total_fee","3_total_fee","4_total_fee","month_traffic","contract_time","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","former_complaint_num","former_complaint_fee"]
    columns = ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee", "month_traffic",
               "pay_num", "last_month_traffic", "local_trafffic_month",
               "local_caller_time", "service1_caller_time", "service2_caller_time"]

    # for feature in columns:
    #     quantileValue = Data[feature].quantile(0.998)
    #     Data[feature][Data[feature]>=quantileValue] = quantileValue
    #     #print(Data[feature].max())

    return Data

def data_preprocess(Data):


    Data["12_total_fee"] = abs(Data["1_total_fee"]- Data["2_total_fee"])
    Data["23_total_fee"] = abs(Data["2_total_fee"] - Data["3_total_fee"])
    Data["34_total_fee"] = abs(Data["3_total_fee"] - Data["4_total_fee"])



    #Data = Data.drop(["2_total_fee", "3_total_fee", "4_total_fee"], axis=1)

    Data["local_month_traffic_dif"] = Data["month_traffic"] - Data["local_trafffic_month"]
    Data["month_traffic>zero"] = (Data["month_traffic"]>1.0).apply(int)
    Data["last_month_traffic>zero"]= (Data["last_month_traffic"]>1.0).apply(int)
    Data["local_trafffic_month>zero"] = (Data["local_trafffic_month"]>1.0).apply(int)




    #Data["onlineTimeRange"] = Data["online_time"].apply(get_onlineTimeRange)
    #dummyFeature = pd.get_dummies(Data["onlineTimeRange"])
    #Data = Data.drop(["online_time","onlineTimeRange"], axis=1)
    #Data = pd.concat([Data, dummyFeature], axis=1)

    Data["mean_service1_2_caller_time"] = (Data["service1_caller_time"]+Data["service2_caller_time"])/2.0
    Data["diff_service1_2_caller_time"] = Data["service2_caller_time"]-Data["service1_caller_time"]

    #Data["complaint_fee/num"] = (Data["former_complaint_fee"]*1.0)/Data["former_complaint_num"]
    #==================New======================
    Data["total_time"] = Data["online_time"]+Data["contract_time"]
    Data["average_total_fee"] = Data[["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]].sum(axis=1)/4.0
    Data["caller_time"] = Data[["local_caller_time", "service1_caller_time", "service2_caller_time"]].sum(axis=1)
    Data["month_traffic>=139"] = (Data["month_traffic"]>=139).apply(int)#.apply(lambda x: int(x>=139))
    Data["contract_time>=0"] = (Data["contract_time"]>=0).apply(int)#.apply(lambda x: int(x >= 0))
    Data["pay_times>=1"] = (Data["pay_times"]>=1).apply(int)#.apply(lambda x: int(x >= 1))
    Data["pay_num>=80"] = (Data["pay_num"]>=80).apply(int)#.apply(lambda x: int(x >= 80))
    Data["last_month_traffic>=0"] = (Data["last_month_traffic"]>=0).apply(int)#.apply(lambda x: int(x >= 0))
    Data["local_trafffic_month>=1262"] = (Data["local_trafffic_month"]>=1262).apply(int)#.apply(lambda x: int(x >= 1262))
    Data["local_caller_time>=14"] = (Data["local_caller_time"]>=14).apply(int)#.apply(lambda x: x >= 14)
    Data["service1_caller_time>=0"] = (Data["service1_caller_time"]>=0).apply(int)#.apply(lambda x: x >= 0)
    Data["service2_caller_time>=29"] = (Data["service2_caller_time"]>=29).apply(int)#.apply(lambda x: x >= 29)
    Data["former_complaint_num>=0"] = (Data["former_complaint_num"]>=0).apply(int)#.apply(lambda x: x >= 0)
    Data["former_complaint_fee>=0"] = (Data["former_complaint_fee"]>=0).apply(int)#.apply(lambda x: x >= 0)


    #===================New+1==========================
    Data["last_month_traffic_dif"] = Data["month_traffic"] - Data["last_month_traffic"]
    Data["local_month_traffic_sum"] = Data["month_traffic"] + Data["local_trafffic_month"]
    Data["online_time>=12"] = (Data["online_time"]>=12).apply(int)#.apply(lambda x: x >= 12)
    Data["online_time>=36"] = (Data["online_time"]>=36).apply(int)#.apply(lambda x: x >= 36)
    Data["14total_fee_dif"] = Data["1_total_fee"] - Data["4_total_fee"]
    Data["13total_fee_dif"] = Data["1_total_fee"] - Data["3_total_fee"]
    billColumns = ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]
    #print(Data[billColumns][Data["many_over_bill"] == 1][:5])
    for item in billColumns:
        Data[item][Data["many_over_bill"] == 1] *=0.9
    #Data[billColumns][Data["many_over_bill"] == 1] = 0#overBillData *0.9
    #print(Data[billColumns][Data["many_over_bill"] == 1][:5])
    #================================================

    # ================Log============================
    Data["month_traffic"] = np.log1p(Data["month_traffic"])
    Data["pay_num"] = np.log1p(Data["pay_num"])
    Data["last_month_traffic"] = np.log1p(Data["last_month_traffic"])
    Data["local_trafffic_month"] = np.log1p(Data["local_trafffic_month"])
    Data["local_caller_time"] = np.log1p(Data["local_caller_time"])
    Data["service1_caller_time"] = np.log1p(Data["service1_caller_time"])
    Data["service2_caller_time"] = np.log1p(Data["service2_caller_time"])
    Data["former_complaint_fee"] = np.log1p(Data["former_complaint_fee"])

    #=========================================================


    return Data





if __name__ == "__main__":
    print("[+]begin")
    train_columns = ["service_type","is_mix_service","online_time","1_total_fee","2_total_fee","3_total_fee","4_total_fee","month_traffic","many_over_bill","contract_type","contract_time","is_promise_low_consume","net_service","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","gender","age","complaint_level","former_complaint_num","former_complaint_fee","current_service","user_id"]
    test_columns = ["service_type", "is_mix_service", "online_time", "1_total_fee", "2_total_fee", "3_total_fee",
                    "4_total_fee", "month_traffic", "many_over_bill", "contract_type", "contract_time",
                    "is_promise_low_consume", "net_service", "pay_times", "pay_num", "last_month_traffic",
                    "local_trafffic_month", "local_caller_time", "service1_caller_time", "service2_caller_time",
                    "gender", "age", "complaint_level", "former_complaint_num", "former_complaint_fee", "user_id"]

    trainData = get_data("data/train.csv",train_columns)
    #print(trainData)
    testData = get_data("data/test.csv",test_columns)
    #print(testData)


    current_service = trainData["current_service"].unique()
    #print(current_service)
    current_service_to_label = {}
    label_to_current_service = {}
    for i in range(15):
        current_service_to_label[current_service[i]]=i
        label_to_current_service[i] = current_service[i]
    #print(current_service_to_label)
    #print(label_to_current_service)

    trainData = trainData.replace({"current_service":current_service_to_label})
    #print(trainData)
    trainLabel = trainData["current_service"].copy()
    #print(trainLabel)
    #trainData = trainData.drop(["user_id","current_service"],axis=1)
    #print(trainData)
#==============================================================================

#==============================================================================
    #trainData[trainData < 0] = 123456
    #print(trainData<0)
    #print(trainData[trainData < 0])
    #print(trainData[trainData["2_total_fee"]=="\\N"])
    two_total_fee_mean = trainData[trainData["2_total_fee"]!="\\N"]["2_total_fee"].astype(float).mean()
    trainData["2_total_fee"].replace("\\N",two_total_fee_mean,inplace=True)
    trainData["2_total_fee"]=trainData["2_total_fee"].astype(float)
    #print(trainData["2_total_fee"])

    three_total_fee_mean = trainData[trainData["3_total_fee"] != "\\N"]["3_total_fee"].astype(float).mean()
    trainData["3_total_fee"].replace("\\N",three_total_fee_mean, inplace=True)
    trainData["3_total_fee"]=trainData["3_total_fee"].astype(float)

    age_mean = trainData[trainData["age"] != "\\N"]["age"].astype(float).mean()
    trainData["age"].replace("\\N", three_total_fee_mean, inplace=True)
    trainData["age"] = trainData["age"].astype(float)

    trainData["gender"].replace("\\N", 0, inplace=True)
    trainData["gender"] = trainData["gender"].astype(float)

    print(trainData.info())



#==============================================================================
    '''
    def get_zs(data):
        #print(pd.value_counts(data["online_time"].tolist()))
        count = pd.value_counts(data["online_time"].tolist())
        #print(count)
        result = np.zeros((1,300))
        #print(count.shape)
        for key,value in count.iteritems():
            #print("[k]%d"%(key))
            #print("[v]%d"%(value))
            result[0][key]=value
        #print(result)
        return result
    
    '''
    '''
    def get_zs1(data):
        #print(pd.value_counts(data["online_time"].tolist()))
        length = data.shape[0]
        count = pd.value_counts(data["current_service"].tolist())
        #print(count)
        result = np.zeros((1,15))
        #print(count.shape)
        for key,value in count.iteritems():
            #print("[k]%d"%(key))
            #print("[v]%d"%(value))
            result[0][key]=value/float(length)
        #print(result)
        return result
    '''
    '''
    #zs = trainData[["online_time","current_service"]].groupby(["current_service"]).apply(get_zs)#lambda x: pd.value_counts(x["online_time"].tolist,sort=False))
    zs = trainData[["online_time","current_service"]].groupby(["online_time"]).apply(get_zs1)
    zs = np.vstack(zs.values)
    #print(np.shape(zs))
    #print(zs[0])
    #print(trainData["online_time"].max())
    #print(trainData["online_time"].min())
    #print(zs)
    #Axes3D.bar(trainData["online_time"],trainData["current_service"],zs=)

    sns.heatmap(zs,cmap="rainbow")
    plt.show()
    '''
    '''
    total_fee = trainData[["4_total_fee","current_service"]]
    result = total_fee.boxplot(by="current_service",sym="",return_type="dict")
    plt.show()
    '''
    '''
    month_traffic = trainData[["month_traffic","current_service"]]
    result = month_traffic.boxplot(by="current_service", return_type="dict", sym="")
    plt.show()
    '''
    '''
    print(trainData["contract_time"].max())
    print(trainData["contract_time"].min())
    print(trainData["contract_time"].unique())

    zs = trainData[["contract_time", "current_service"]].groupby(["contract_time"]).apply(get_zs1)
    zs = np.vstack(zs.values)

    sns.heatmap(zs, cmap="rainbow")
    plt.show()
    '''
    '''
    zs = trainData[["net_service", "current_service"]].groupby(["net_service"]).apply(get_zs1)
    zs = np.vstack(zs.values)

    sns.heatmap(zs, cmap="rainbow")
    plt.show()
    '''
    '''
    print(trainData["pay_times"].max())
    print(trainData["pay_times"].min())
    print(np.sort(trainData["pay_times"].unique()))
    zs = trainData[["pay_times", "current_service"]].groupby(["pay_times"]).apply(get_zs1)
    zs = np.vstack(zs.values)

    sns.heatmap(zs, cmap="rainbow")
    plt.show()
    '''
    '''
    pay_num = trainData[["pay_num", "current_service"]]
    result = pay_num.boxplot(by="current_service", return_type="dict", sym="")
    plt.show()
    '''
    '''
    last_month_traffic = trainData[["last_month_traffic", "current_service"]]
    result = last_month_traffic.boxplot(by="current_service", return_type="dict", sym="")
    plt.show()
    '''
    '''
    trainData["local_traffic_rate"] = trainData["local_trafffic_month"]/(trainData["month_traffic"]+trainData["last_month_traffic"])
    local_traffic_rate = trainData[["local_traffic_rate", "current_service"]]
    result = local_traffic_rate.boxplot(by="current_service", return_type="dict", sym="")
    plt.yticks(np.arange(0, 2, step=0.2))
    plt.show()
    '''
    '''
    local_caller_time = trainData[["local_caller_time", "current_service"]]
    result = local_caller_time.boxplot(by="current_service", return_type="dict", sym="")
    #plt.yticks(np.arange(0, 2, step=0.2))
    plt.show()
    '''
    '''
    service1_caller_time = trainData[["service1_caller_time", "current_service"]]
    result = service1_caller_time.boxplot(by="current_service", return_type="dict", sym="")
    plt.yticks(np.arange(0, 160, step=5))
    plt.show()
    '''
    '''
    service2_caller_time = trainData[["service2_caller_time", "current_service"]]
    result = service2_caller_time.boxplot(by="current_service", return_type="dict", sym="")
    plt.yticks(np.arange(0, 2000, step=20))
    plt.show()
    '''
    '''
    print(trainData["age"].max())
    print(trainData["age"].min())
    print(np.sort(trainData["age"].unique()))
    zs = trainData[["age", "current_service"]].groupby(["age"]).apply(get_zs1)
    zs = np.vstack(zs.values)

    sns.heatmap(zs, cmap="rainbow")
    plt.show()
    '''
    '''
    zs = trainData[["complaint_level", "current_service"]].groupby(["complaint_level"]).apply(get_zs1)
    zs = np.vstack(zs.values)

    sns.heatmap(zs, cmap="rainbow")
    plt.show()
    '''
    '''
    print(trainData["former_complaint_num"].max())
    print(trainData["former_complaint_num"].min())
    print(np.sort(trainData["former_complaint_num"].unique()))
    former_complaint_num = trainData[["former_complaint_num", "current_service"]]
    result = former_complaint_num.boxplot(by="current_service", return_type="dict", sym="")
    plt.show()
    '''
    '''
    #print(trainData["former_complaint_fee"].max())
    #print(trainData["former_complaint_fee"].min())
    #print(np.sort(trainData["former_complaint_fee"].unique()))
    #former_complaint_num = trainData[trainData["former_complaint_fee"]!=0][["former_complaint_fee", "current_service"]]
    former_complaint_num = trainData[["former_complaint_fee", "current_service"]]
    rate = former_complaint_num.groupby(["current_service"]).apply(lambda x:x[x["former_complaint_fee"]!=0].shape[0]/float(x.shape[0]))
    print(rate)
    plt.plot(range(15),rate)
    plt.title("former_complaint_fee>0 rate")
    plt.grid(axis="y")
    for i in range(15):
            plt.text(i, rate[i] + 0.001, '%.4f' % rate[i], ha='center', va='bottom', fontsize=9)
    #result = former_complaint_num.boxplot(by="current_service", return_type="dict", sym="")
    plt.show()
    '''
    '''
    modeData = trainData[["service_type","is_mix_service","many_over_bill","contract_type","is_promise_low_consume","net_service","gender","complaint_level","current_service"]].groupby("current_service").apply(lambda x: x.mode())
    modeData = modeData.drop("current_service",axis=1).reset_index()
    print(modeData.info())

    middleData = trainData[["online_time","1_total_fee","2_total_fee","3_total_fee","4_total_fee","month_traffic","contract_time","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","age","former_complaint_num","former_complaint_fee","current_service"]].groupby(["current_service"]).apply(lambda x: x.median())
    middleData = middleData.drop("current_service",axis=1).reset_index()
    print(middleData.info())

    serviceData = pd.concat([modeData,middleData],axis=1)
    print(serviceData)

    for i in serviceData.columns:
        serviceData[i].plot(xticks=range(15))
        plt.title(i)
        plt.show()
    #serviceData.plot(subplots=True,xticks=range(15))

    #plt.legend()
    #plt.show()
    '''
    '''
    zeroCount = trainData[trainData["gender"]==0].groupby("current_service").apply(lambda x: x.shape[0])
    zeroRate = trainData.groupby("current_service").apply(lambda x: float(x.shape[0]))
    zeroRate = round(zeroCount/zeroRate,4)
    print(list(zeroRate))
    '''
    '''
    rate = {}
    uni = trainData["gender"].unique()
    all = trainData.shape[0]
    for i in uni:
        count = trainData[trainData["gender"]==i].shape[0]
        rate[i] = round(count/float(all),4)
    print(rate)
    '''
    trainData["average_total_fee"] = trainData[["1_total_fee", "2_total_fee", "3_total_fee"]].sum(axis=1)
    print(trainData["average_total_fee"])
    #print(trainData[["average_total_fee","contract_time","pay_times","pay_num","last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time","service2_caller_time","former_complaint_num","former_complaint_fee"]].describe())










