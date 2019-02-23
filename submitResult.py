# -*- coding:utf-8 -*-
from Layers import *

def get_curr2label():
    trainData = get_data("data/processed/trainData.csv")
    print("[+]Load TrainData Successed!")
    current_service = trainData["current_service"].unique()
    current_service = pd.DataFrame(current_service)
    current_service.to_csv("current_service_unique.csv",index=False)
    exit()


if __name__ =="__main__":
    print("[+]Begin....")

    result1 = get_data("result_xgb_3_1.csv")
    print(np.shape(result1))
    print("[+]Load result_xgb_3_1 Successed!")

    result2 = get_data("result_lgb_3_2.csv")
    print(np.shape(result2))
    print("[+]Load result_lgb_3_2 Successed!")

    result3 = get_data("result_lgb_3_3.csv")
    print(np.shape(result3))
    print("[+]Load result_lgb_3_3 Successed!")

    testData = get_data("data/processed/testData.csv")
    print("[+]Load TestData Successed!")

    current_service = get_data("current_service_unique.csv")["0"].tolist()

    label_to_current_service = dict(zip(range(0, len(current_service)), current_service))
    current_service_to_label = dict(zip(current_service, range(0, len(current_service))))

    result = pd.concat([pd.DataFrame(result1),pd.DataFrame(result2),pd.DataFrame(result3)],axis=1)
    result = result.applymap(lambda x:current_service_to_label[x])
    #print(result[:5])
    print(result.shape)


    submit = []
    for index,row in result.iterrows():
        submit.append(np.argmax(np.bincount(row)))
    df_test = pd.DataFrame()
    df_test['id'] = testData["user_id"].copy()
    df_test['predict'] = submit
    df_test['predict'] = df_test['predict'].map(label_to_current_service)
    print(df_test[:5])
    print(df_test.shape)
    df_test.to_csv("allResult.csv", index=False)
    print("[+]Done!")