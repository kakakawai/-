# CCF-Dianxin
CCF数据挖掘-面向电信行业存量用户的智能套餐个性化匹配模型-初赛top10%
https://www.datafountain.cn/competitions/311/details
## 简介
![image](https://github.com/kakakawai/CCF-Dianxin/blob/master/figure/%E6%A8%A1%E5%9E%8B.png)

Layer1
* 原始训练集->xgb->trainData_xgb1 | testData_xgb1 | result_xgb1

Layer2
* 原始训练集+xgb1 -> lgb -> trainData_lgb_2_1 | testData_lgb_2_1 | result_lgb_2_1

Layer3
* lgb_3_1:原始训练集+lgb_2_1 -> xgb -> trainData_xgb_3_1 | testData_xgb_3_1 | result_xgb_3_1
* lgb_3_2:原始训练集 -> lgb -> trainData_lgb_3_2 | testData_lgb_3_2 | result_lgb_3_2
* lgb_3_3:原始训练集+xgb1 -> lgb -> trainData_lgb_3_3 | testData_lgb_3_3 | result_lgb_3_3

## 使用
```
./oneForAll.sh
```
