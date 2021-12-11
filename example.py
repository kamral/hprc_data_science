import  pandas as pd

X_full=pd.read_csv('train.csv')
print(X_full) #[1460 rows x 81 columns]
X_full_test=pd.read_csv('test.csv')
print(X_full_test) #[1459 rows x 80 columns]


X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
print(X_full) #[1460 rows x 81 columns]
y=X_full.SalePrice
print(y) #Name: SalePrice, Length: 1460, dtype: int64

X_full.drop(['SalePrice'],axis=1, inplace=True)
print(X_full) #[1460 rows x 80 columns]


X=X_full.select_dtypes(exclude=['object'])
print(X) #[1460 rows x 37 columns]
X_test=X_full_test.select_dtypes(exclude=['object'])
print(X_test) #[1459 rows x 37 columns]

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid=train_test_split(X,y,train_size=0.7,
                                                    test_size=0.3,random_state=1)
print(X_train) #[1021 rows x 37 columns]

print(X_valid) #[438 rows x 37 columns]

##########################


column_without_Nan=[col for col in X_train.columns
                    if X_train[col].isnull().any()]

reduced_X_train=X_train.drop(column_without_Nan, axis=1)
print(reduced_X_train) #[1021 rows x 34 columns]

reduced_X_valid=X_valid.drop(column_without_Nan, axis=1)
print(reduced_X_valid) #[438 rows x 34 columns]

#################################################################

from sklearn.impute import SimpleImputer
final_imputer=SimpleImputer(strategy='median')

final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

final_X_train.columns=X_train.columns
final_X_valid.columns=X_valid.columns



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error



def score_dataset(X_train, X_valid, y_train, y_valid):
    model=RandomForestClassifier(random_state=1)
    model.fit(X_train,y_train)
    preds=model.predict(X_valid)
    return mean_absolute_error(y_valid,preds)

print(score_dataset(final_X_train,final_X_valid,y_train,y_valid))
#24789.461187214612
###########################################################################
###########################################################################

# from sklearn.ensemble import RandomForestRegressor
#
# model = RandomForestRegressor(n_estimators=100, random_state=0)
# model.fit(final_X_train, y_train)
#
# # Get validation predictions and MAE
# preds_valid = model.predict(final_X_valid)
# print("MAE (Your approach):")
# print(mean_absolute_error(y_valid, preds_valid)) #MAE (Your approach):17715.0253196347





#######################################################

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))

final_X_test = pd.DataFrame(final_imputer.transform(X_test))


# # Get test predictions
preds_test = model.predict(final_X_test)
# # MAE (Your approach):17715.0253196347