from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

data=pd.read_csv('train.csv')
print(data) #[1460 rows x 81 columns]
data_test=pd.read_csv('test.csv')
print(data_test)#[1459 rows x 80 columns]
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
print(data) #[1460 rows x 81 columns]

y=data.SalePrice
X=data.select_dtypes(exclude=['object'])
print(X) #[1460 rows x 38 columns]

X=X.drop(['SalePrice'],axis=1)
print(X)

X_test=data_test.select_dtypes(exclude=['object'])
print(X_test) #[1459 rows x 37 columns]

print(y) #Name: SalePrice, Length: 1460, dtype: int64

X_train, X_valid, y_train,y_valid=train_test_split(X,y, train_size=0.7,
                                                   test_size=0.3, random_state=1)
print(X_train) #[1021 rows x 290 columns]
#[1021 rows x 38 columns]

print(X_valid) #[438 rows x 290 columns]
#[438 rows x 38 columns]


imputer=SimpleImputer()
imputer_X_train=pd.DataFrame(imputer.fit_transform(X_train))
imputer_X_valid=pd.DataFrame(imputer.transform(X_valid))

imputer_X_train.columns=X_train.columns
imputer_X_valid.columns=X_valid.columns

clf=RandomForestRegressor(random_state=1)
clf.fit(imputer_X_train,y_train)
preds=clf.predict(imputer_X_valid)
mean_absolute_errors=mean_absolute_error(y_valid,preds)
print(mean_absolute_errors)
# 864.3357762557079

final_X_test=pd.DataFrame(imputer.transform(X_test))
preds=clf.predict(final_X_test)

output=pd.DataFrame({
    'Id':X_test.index,
    'SalePrice':preds
})

output.to_csv('Submission.csv', index=False)













