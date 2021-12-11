import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


data_auto=pd.read_csv('Automobile_data.csv')
print(data_auto)
'''

     symboling normalized-losses         make fuel-type aspiration  ... horsepower peak-rpm city-mpg highway-mpg  price
0            3                 ?  alfa-romero       gas        std  ...        111     5000       21          27  13495
1            3                 ?  alfa-romero       gas        std  ...        111     5000       21          27  16500
2            1                 ?  alfa-romero       gas        std  ...        154     5000       19          26  16500
3            2               164         audi       gas        std  ...        102     5500       24          30  13950
4            2               164         audi       gas        std  ...        115     5500       18          22  17450
..         ...               ...          ...       ...        ...  ...        ...      ...      ...         ...    ...
200         -1                95        volvo       gas        std  ...        114     5400       23          28  16845
201         -1                95        volvo       gas      turbo  ...        160     5300       19          25  19045
202         -1                95        volvo       gas        std  ...        134     5500       18          23  21485
203         -1                95        volvo    diesel      turbo  ...        106     4800       26          27  22470
204         -1                95        volvo       gas      turbo  ...        114     5400       19          25  22625

'''

y=data_auto.pop('price')
print(y)
print(data_auto.dtypes)

for colname in data_auto.select_dtypes("object"):
    data_auto[colname], _ = data_auto[colname].factorize()

print(data_auto)
'''
symboling  normalized-losses  make  fuel-type  aspiration  ...  compression-ratio  horsepower  peak-rpm  city-mpg  highway-mpg
0            3                  0     0          0           0  ...                9.0           0         0        21           27
1            3                  0     0          0           0  ...                9.0           0         0        21           27
2            1                  0     0          0           0  ...                9.0           1         0        19           26
3            2                  1     1          0           0  ...               10.0           2         1        24           30
4            2                  1     1          0           0  ...                8.0           3         1        18           22
..         ...                ...   ...        ...         ...  ...                ...         ...       ...       ...          ...
200         -1                 51    21          0           0  ...                9.5          56         4        23           28
201         -1                 51    21          0           1  ...                8.7           6        23        19           25
202         -1                 51    21          0           0  ...                8.8          58         1        18           23
203         -1                 51    21          1           1  ...               23.0          59         6        26           27
204         -1                 51    21          0           1  ...                9.5          56         4        19           25

'''

