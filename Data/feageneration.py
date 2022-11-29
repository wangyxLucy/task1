import pandas as pd
import numpy as np

def transform(data, path):
    types = np.unique(data['type'].values)
    count = 0
    newcol = np.zeros(data.shape[0])
    flag = data.loc[0, 'type']
    count = 0
    for i in range(1,data.shape[0]):
        if data.loc[i, 'type'] == flag:
            newcol[i] = count
        else:
            count = count + 1
            newcol[i] = count
        flag = data.loc[i, 'type']
    data['newcol'] = newcol
    data.to_csv(path)

data = pd.read_csv('Data/HK_demand_modeling_input_level_2train.csv')
transform(data, 'Data/new_train.csv')
data = pd.read_csv('Data/HK_demand_modeling_input_level_2test.csv')
transform(data, 'Data/new_test.csv')



    
        