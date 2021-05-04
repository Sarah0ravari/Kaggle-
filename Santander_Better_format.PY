
import numpy as np

#from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
#from sklearn.metrics import accuracy_score

#Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

#For Missing Value and Feature Engineering
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier

import time

#import the data
train = pd.read_csv(train)
test = pd.read_csv(test)

train.head()

##Divide the data to X and Y
#create X and y datasets for splitting 
X = train.drop(['ID', 'TARGET'], axis=1)

all_features = X.columns

all_features = all_features.tolist()

numerical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind in ['i','f'] and c !='PassengerId']
categorical_features = [c for c, dtype in zip(X.columns, X.dtypes)
                     if dtype.kind not in ['i','f']]
                     
numerical_features

categorical_features

#import train_test_split library
from sklearn.model_selection import train_test_split

# create train test split
X_train, X_test, y_train, y_test = train_test_split( X,  y, test_size=0.3, random_state=0) 

##Setup Pipeline
preprocessor = make_column_transformer(
    
    (make_pipeline(
    #SimpleImputer(strategy = 'median'),
    KNNImputer(n_neighbors=2, weights="uniform"),
    MinMaxScaler()), numerical_features),
    
    (make_pipeline(
    SimpleImputer(strategy = 'constant', fill_value = 'missing'),
    OneHotEncoder(categories = 'auto', handle_unknown = 'ignore')), categorical_features),
    
)

preprocessor_best = make_pipeline(preprocessor, 
                                  VarianceThreshold(), 
                                  SelectKBest(f_classif, k = 15), 
                                  PCA(n_components = 4))
# boosting tree models which is best gradient performance                                   
XGB_Model = make_pipeline(preprocessor_best, XGBClassifier())

#grid  search 
# Create the param grid - xgbclassifier
param_grid = {
 "xgbclassifier__learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "xgbclassifier__max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "xgbclassifier__min_child_weight" : [ 1, 3, 5, 7 ],
 "xgbclassifier__gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "xgbclassifier__colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   
}

from sklearn.model_selection import RandomizedSearchCV
xgb_RandomGrid = RandomizedSearchCV(estimator = XGB_Model, 
param_distributions = param_grid, cv = 5, verbose=3, n_jobs = -1,
scoring = 'roc_auc', n_iter = 5)
xgb_RandomGrid.fit(X_train, y_train)

xgb_RandomGrid.best_estimator_

## Acurracy 
print(f'Train : {xgb_RandomGrid.score(X_train, y_train):.3f}')
print(f'Test : {xgb_RandomGrid.score(X_test, y_test):.3f}')

from sklearn.metrics import roc_auc_score

print(f'Train AUC : {roc_auc_score(y_train, xgb_RandomGrid.predict_proba(X_train)[:,1]):.3f}')
print(f'Train AUC : {roc_auc_score(y_test, xgb_RandomGrid.predict_proba(X_test)[:,1]):.3f}')

#submission
test_pred = xgb_RandomGrid.predict_proba(test[X.columns])[:,1]
#test_pred = rf_RandomGrid.predict(test[X.columns])
AllSub = pd.DataFrame({ 'ID': test['ID'],
                       'TARGET' : test_pred
    
})

#AllSub['TARGET'] = AllSub['TARGET'].apply(lambda x: 1 if x > 0.5 else 0)

AllSub.to_csv('Santander_XGB_Better_Pipe.csv', index = False)
