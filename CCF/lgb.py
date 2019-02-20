import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

#data pre-processing

train = train[train.gender != '\\N']
# test = test[test.gender != '\\N']
train['gender'] = train['gender'].apply(lambda x : int(x))
test['gender'] = test['gender'].apply(lambda x : int(x))

train = train[train.age != '\\N']
# test = test[test.age != '\\N']
train['age'] = train['age'].apply(lambda x : int(x))
test['age'] = test['age'].apply(lambda x : int(x))
'''
train = train[train.service_type != '\\N']
# test = test[test.service_type != '\\N']
train['service_type'] = train['service_type'].apply(lambda x : int(x))
test['service_type'] = test['service_type'].apply(lambda x : int(x))
'''
train = train[train['2_total_fee'] != '\\N']
# test = test[test['2_total_fee'] != '\\N']
test.loc[test['2_total_fee'] == '\\N','2_total_fee'] = 0.0
train['2_total_fee'] = train['2_total_fee'].apply(lambda x : float(x))
test['2_total_fee'] = test['2_total_fee'].apply(lambda x : float(x))

train = train[train['3_total_fee'] != '\\N']
# test = test[test['3_total_fee'] != '\\N']
test.loc[test['3_total_fee'] == '\\N','3_total_fee'] = 0.0
train['3_total_fee'] = train['3_total_fee'].apply(lambda x : float(x))
test['3_total_fee'] = test['3_total_fee'].apply(lambda x : float(x))


label = train.pop('current_service')
le = LabelEncoder()
label = le.fit_transform(label)


feature = [value for value in train.columns.values if
                   value not in ['user_id']]


#lgb model
def LGB():
        clf = lgb.LGBMClassifier(
                bjective='multiclass',
                boosting_type='gbdt',
                num_leaves=80,
                max_depth=7,
                min_data_in_leaf=300,
                learning_rate=0.05,
                lambda_l1=0.1,
                lambda_l2=0.2,
                seed=2018,
                colsample_bytree=0.8,
                subsample=0.9,
                n_estimators=20)  #2000:0.740205;3500:0.7390;2600:73943;2200:73938;1800:740205;1900:74018
        return clf

#online = False
online = True # please '# online = False'if you would like to submit
if online:
        print ('online')

        model = LGB()
        model.fit(train[feature], label, eval_set=[(train[feature], label)], verbose=1)
        pred = model.predict(test[feature])
        pred = le.inverse_transform(pred)
        test['predict'] = pred

        test[['user_id', 'predict']].to_csv('./data/result2.csv', index=False)
else:
        print ('offline')
        train_x,test_x,train_y,test_y = train_test_split(train[feature],label,test_size=0.1,shuffle=True,random_state=2018)
        model = LGB()
        model.fit(train_x[feature], train_y, eval_set=[(test_x[feature], test_y)], verbose=1,early_stopping_rounds=100)
        pred = model.predict(test_x)
        print(f1_score(test_y,pred,average='weighted'))

