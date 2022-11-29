import pandas as pd
import datetime
import numpy as np
import torch


"""
Split features and label
"""
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def transform(df):

    target = np.array(df['y'])
    date = np.array(df['time'])
    SKU = np.array(df['type'])

    numofkinds = len(np.unique(SKU))
    numofday = len(np.unique(date))

    date = np.unique(date)
    target = np.reshape(target, [numofkinds, numofday])

    df_output = pd.DataFrame(target)
    df_output.columns = date

    return df_output

"""
Data Preprocessing: Transform data into specific format 
"""
def DataLoading(data):
    Dataset = transform(data)

    TOTALDAYNUM = Dataset.shape[1]
    TOTALDEVICENUM = Dataset.shape[0]

    Dataset = Dataset.transpose()

    #=============================================================
    # Rename the columns in printerDataset_Mono_Transpose
    #=============================================================

    Dataset.index.name = 'Datetime'
    Dataset.reset_index(inplace=True)
    
    for colIdx in range( TOTALDEVICENUM ):
        tmpColName  = 'Dev' + str( Dataset.columns[ colIdx+1 ]  )
        Dataset.rename(columns = {Dataset.columns[ colIdx + 1 ]:tmpColName}, inplace = True)
    date = Dataset.iloc[:,0].values
    return Dataset, date

"""
Functions computing evaluation metrics
"""

def masked_mse(preds, labels, null_val=np.nan):
    loss = (preds.float()-labels.float())**2
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    loss = torch.abs(preds.float()-labels.float())
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    loss = torch.abs(preds.float()-labels.float())/labels.float()
    loss = loss * mask.float()
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,rmse,mape



