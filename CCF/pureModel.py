import pandas as pd
import lightgbm as lgb
import gc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

current_service_series = [89950166, 89950167, 89950168, 90063345, 90109916, 90155946, 99999825, 99999826, 99999827,
                              99999828, 99999830]
constant_feats=['1_total_fee','2_total_fee','3_total_fee','4_total_fee','month_traffic','pay_times','pay_num','last_month_traffic','local_trafffic_month','local_caller_time','service1_caller_time','service2_caller_time','age','former_complaint_num','former_complaint_fee']
category_feats=['service_type','is_mix_service','online_time','many_over_bill','contract_type','contract_time','is_promise_low_consume','net_service','complaint_level','gender']
#输出准确率
def do_model_metric(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import roc_auc_score,accuracy_score
    #print("AUC: {0:.3}".format(roc_auc_score(y_true=y_true, y_score=y_pred_prob[:,1])))
    print("Accuracy: {0}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))


#current_service的values转化为0-11
def changeValues(current_service):
    for k in range(len(current_service)):
        for i in range(11):
            if (current_service[k] == current_service_series[i]):
                current_service[k] = i
    return current_service


#读取数据
train_data=pd.read_csv("./data/train.csv")
test_data=pd.read_csv("./data/test.csv")
train_sz = train_data.shape[0]#记录train有多少行
current_service =train_data.current_service.values
user_id=train_data.user_id.values
current_service=changeValues(current_service)
train_data.drop(columns=['user_id','current_service'],inplace=True)
test_data.drop(columns=['user_id'],inplace=True)
combine_data=pd.concat([train_data, test_data], axis=0, ignore_index=True)#train和test数据连接，之后一起处理脏数据和连续数据归一化和离散数据one_hot
del test_data,train_data
gc.collect()

#处理脏数据
#age
age=combine_data.age.values
for i in range(len(combine_data.age)):
    if(age[i]=='\\N'):
        age[i]=0
    else:
        age[i]=int(age[i])
combine_data.age=age
del age
gc.collect()
#gender
gender=combine_data.gender.values
for i in range(len(combine_data.gender)):
    if(gender[i]==1):
        gender[i]=1
    elif(gender[i]==2):
        gender[i]=2
    else:
       gender[i]=0
combine_data.gender=gender
del gender
gc.collect()
#2_total_fee
two_total_fee=combine_data['2_total_fee'].values
for i in range(len(combine_data['2_total_fee'])):
    if(two_total_fee[i]=='\\N'):
        two_total_fee[i]=float(0)
    else:
        two_total_fee[i]=float(two_total_fee[i])
combine_data['2_total_fee']=two_total_fee
del two_total_fee
gc.collect()
#3_total_fee
three_total_fee=combine_data['3_total_fee'].values
for i in range(len(combine_data['3_total_fee'])):
    if(three_total_fee[i]=='\\N'):
        three_total_fee[i]=float(0)
    else:
        three_total_fee[i]=float(three_total_fee[i])
combine_data['3_total_fee']=three_total_fee
del three_total_fee
gc.collect()

#连续数据归一化
for col in constant_feats:
    scaler = MinMaxScaler()
    combine_data[col] = scaler.fit_transform(np.array(combine_data[col].values.tolist()).reshape(-1,1))

#离散数据one_hot
#print(combine_data.info())
for col in category_feats:
    onehot = pd.get_dummies(combine_data[col], prefix=col)
    combine_data= pd.concat([combine_data, onehot], axis=1)
combine_data.drop(columns=category_feats,inplace=True)

#train和test数据分离
train_data=combine_data[:train_sz]
test_data=combine_data[train_sz:]
lgb_feats=train_data.columns.values.tolist()

#训练GBDT
X_train,X_validation,y_train,y_validation=train_test_split(train_data,current_service,test_size=0.4)
train_data=lgb.Dataset(train_data,label=current_service)
validation_data=lgb.Dataset(X_validation,label=y_validation)
del X_train,X_validation,y_train,y_validation
gc.collect()
params={
    'learning_rate':0.02,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':5,
    'num_leaves':10,
    'objective':'multiclass',
    'num_class':11,
}

newCategory_feats = [col for col in combine_data.columns if col not in constant_feats]
clf=lgb.train(params,train_data,valid_sets=[validation_data],categorical_feature=newCategory_feats)
del validation_data,train_data
gc.collect()

#得到GBDT后的叶节点编号 ，并将其one_hot并入整个数据集
gbdt_feats_vals = clf.predict(combine_data, pred_leaf=True)
gbdt_columns = ["gbdt_leaf_indices_" + str(i) for i in range(0, gbdt_feats_vals.shape[1])]
combine_data = pd.concat([combine_data, pd.DataFrame(data=gbdt_feats_vals, index=range(0, gbdt_feats_vals.shape[0]),columns=gbdt_columns)], axis=1)
origin_columns = combine_data.columns
for col in gbdt_columns:
    combine_data = pd.concat([combine_data, pd.get_dummies(combine_data[col], prefix=col)],axis=1)
gbdt_onehot_feats = [col for col in combine_data.columns if col not in origin_columns]

# 恢复train, test
train = combine_data[:train_sz]
test = combine_data[train_sz:]
del combine_data
gc.collect()

#训练
lr_gbdt_feats = lgb_feats+ gbdt_onehot_feats
lr_gbdt_model = LogisticRegression(penalty='l2', C=1,multi_class='multinomial')
print("Train................")
lr_gbdt_model.fit(train[lr_gbdt_feats],current_service)

#测试test,输出
print("Test..................")
y_pred=lr_gbdt_model.predict(test[lr_gbdt_feats])
for k in range(len(y_pred)):
    for i in range(11):
        if(y_pred[k]==i):
            y_pred[k]=current_service_series[i]
print(y_pred)
#print(accuracy_score(y_test,y_pred))
result={'user_id':user_id,'current_service':y_pred}
resultDf=pd.DataFrame(result)
resultDf.to_csv("./data/submit1.csv",index=False)

