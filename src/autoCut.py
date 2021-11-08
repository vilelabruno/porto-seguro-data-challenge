# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb

from geopy.distance import geodesic 
df_test = pd.read_csv('../data/test.csv')


# %%




# %%



# %%



# %%


# %%

# %%

doc = {}
for col2 in range(25,69): # 100
    col2 = "var"+str(col2)
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    target_train = df_train['y'].values
    id_test = df_test['id'].values
    
    
    # %%
    
    
    
    
    # %%
    
    
    
    # %%
    
    
    
    # %%
    dummies = 20
    for col in df_test.columns: # 100
        if (len(df_test[col].unique()) < dummies) and (len(df_test[col].unique()) == len(df_train[col].unique())):
            df_train = pd.get_dummies(df_train, columns=[col])
            df_test = pd.get_dummies(df_test, columns=[col])
    try:
        df_train[col2], _ = pd.factorize(pd.cut(df_train[col2], 4))
        df_test[col2], _ =  pd.factorize(pd.cut(df_test[col2], 4))
    except:
        continue
    train = np.array(df_train.drop(['y','id'], axis = 1))
    test = np.array(df_test.drop(['id'], axis = 1))

    xgb_preds = []


# %%
    K = 5
    kf = KFold(n_splits = K, random_state = 3228, shuffle = True)


# %%
    from sklearn.metrics import f1_score
    from sklearn.metrics import log_loss
    def xgb_f1(y, t, threshold=0.5):
        t = t.get_label()
        y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
        return 'f1',f1_score(t,y_bin)
    err = 0
    err2 = 0
    err3 = 0
    err4 = 0
    err5 = 0
    errLL = 0
    cutPoint = 0.36
    for train_index, test_index in kf.split(train):
        train_X, valid_X = train[train_index], train[test_index]
        train_y, valid_y = target_train[train_index], target_train[test_index]

        # params configuration also from the1owl's kernel
        # https://www.kaggle.com/the1owl/forza-baseline
        xgb_params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
        #xgb_params = {'eta': 0.11, 'max_depth': 5, 'subsample': 0.4, 'colsample_bytree': 0.4, 'objective': 'binary:logistic', 'min_child_weight': 15, 'eval_metric': 'auc', 'seed': 99, 'silent': True}

        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)
        d_test = xgb.DMatrix(test)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(xgb_params, d_train, 5000,  watchlist, maximize=True, verbose_eval=50, early_stopping_rounds=100)

        xgb_pred = model.predict(d_test)
        pG = model.predict(d_valid)
        arr = []
        errLL += log_loss(valid_y, pG)
        
        p = pG.copy()
        p[p > cutPoint] = 1
        p[p != 1] = 0
        err += f1_score(valid_y, p)
        p = pG
        p[p > (cutPoint+0.01)] = 1
        p[p != 1] = 0
        err2 += f1_score(valid_y, p)
        p = pG
        p[p > (cutPoint+0.02)] = 1
        p[p != 1] = 0
        err3 += f1_score(valid_y, p)
        p = pG
        p[p > (cutPoint-0.02)] = 1
        p[p != 1] = 0
        err4 += f1_score(valid_y, p)
        p = pG
        p[p > (cutPoint-0.01)] = 1
        p[p != 1] = 0
        err5 += f1_score(valid_y, p)
        print(err)
        print(model.get_score(importance_type='gain'))
        xgb_preds.append(list(xgb_pred))

    doc[col2] = [errLL/5, err/5, err2/5, err3/5, err4/5, err5/5]
    

# %%
df_train # apriori, analise de fator
df_train # apriori, analise de fator


# %%
print(err/5) #print(err/4) #6711266
print(err2/5) #print(err/4) #6711266
print(err3/5) #print(err/4) #6711266
print(err4/5) #print(err/4) #6711266
print(err5/5) #print(err/4) #6711266
print(errLL/5) #print(err/4) #6711266


# %%
preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(K):
        sum+=xgb_preds[j][i]
    preds.append(sum / K)

output = pd.DataFrame({'id': id_test, 'predicted': preds})

output.to_csv("../data/output/proba/{}-foldCV_avg_sub_dummy_dist_cutpoint{}_error{}_logloss{}.csv".format(K,cutPoint, err/5, errLL/5), index=False)   
output['predicted'][output['predicted'] > cutPoint] = 1
output['predicted'][output['predicted'] != 1] = 0
output['predicted'] = output['predicted'].astype(int)
output.to_csv("../data/output/{}-foldCV_avg_sub_dummy_dist_cutpoint{}_error{}_logloss{}.csv".format(K,cutPoint, err/5, errLL/5), index=False)   


# %%
sub = pd.read_csv("./5-foldCV_avg_sub_36_dummy.csv")


# %%
sub["v"] = output["predicted"]


# %%
sub["v2"] = sub["v"] - sub["predicted"]


# %%
sub[sub["v2"] != 0]


# %%
119


# %%



