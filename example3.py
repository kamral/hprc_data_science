from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = pd.read_csv('train.csv', index_col='Id')
print(X) #[1460 rows x 80 columns]
X_test = pd.read_csv('test.csv', index_col='Id')
print(X_test) #[1459 rows x 79 columns]

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
print(X) #[1460 rows x 80 columns]
y=X.SalePrice
print(y)  #Name: SalePrice, Length: 1460, dtype: int64
X.drop(['SalePrice'], axis=1, inplace=True)
print(X) #[1460 rows x 79 columns]
column_without_Nan=[col for col in X.columns
                   if X[col].isnull().any()]
print(column_without_Nan)
'''
['LotFrontage', 'Alley', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
'''
X.drop(column_without_Nan, axis=1, inplace=True)
print(X) #[1460 rows x 60 columns]
X_test.drop(column_without_Nan, axis=1, inplace=True)
print(X_test) #[1459 rows x 60 columns]
X_train, X_valid, y_train, y_valid=train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train) #[1168 rows x 60 columns]
print(X_valid) #[292 rows x 60 columns]

basic_cols_with_categorial_variables=[col for col in X_train.columns
                                     if X_train[col].dtype=='object']
print(basic_cols_with_categorial_variables)
'''
['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir',
'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
'''

good_label_cols=[col for col in basic_cols_with_categorial_variables
                                    if set(X_valid[col]).issubset(set(X_train[col]))]
print(good_label_cols)
'''
['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 
'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition']
'''

bad_label_cols=list(set(basic_cols_with_categorial_variables)-set(good_label_cols))
print(bad_label_cols)
'''
['Condition2', 'RoofMatl', 'Functional']
'''

s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

label_X_train=X_train.drop(bad_label_cols, axis=1)
print(label_X_train)#[1168 rows x 57 columns]
label_X_valid=X_valid.drop(bad_label_cols, axis=1)
print(label_X_valid) #[292 rows x 57 columns]
label_X_test=X_test.drop(bad_label_cols, axis=1)
print(label_X_valid) #[292 rows x 57 columns]



order_encoding=OrdinalEncoder()
label_X_train[good_label_cols]=order_encoding.fit_transform(label_X_train[good_label_cols])
print(label_X_train) #[1168 rows x 57 columns]
label_X_valid[good_label_cols]=order_encoding.transform(label_X_valid[good_label_cols])
print(label_X_valid) #[292 rows x 57 columns]
# label_X_test[good_label_cols]=order_encoding.transform(label_X_test[good_label_cols])
# print(label_X_test) #[292 rows x 57 columns]

# def score_dataset(label_X_train, label_X_valid):
#     model



def score_dataset(label_X_train,X_test, y_train, y_valid):
    model=RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(label_X_train, y_train)
    preds=model.predict(X_test)
    return mean_absolute_error(y_valid, preds)
print(score_dataset(label_X_train, label_X_valid, y_train,y_valid)) #17098.01649543379


