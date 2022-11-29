from GNN.Setting.setting import *

from GNN.Model_Construction.GNN import *

from GNN.Model_Construction.model_run import *
from GNN.Data_Preparation.feature import *
from GNN.Data_Preparation.dataloading import *
from GNN.Data_Preparation.template import *
import pandas as pd
import numpy as np


args = set()
val_ratio = args.val_ratio

def Test(train_data, test_data, loss, trained, save_name):

    """
    Load data 
    """

    #按时间排序


    #取第一天的，看哪些量的unique value不为1，然后记录一下他的名字
    special_var = []
    data0 = train_data[train_data['time']==train_data['time'].values[0]].values
    for i in range(train_data.shape[1]):
        col = data0[:,i]
        if (len(np.unique(col))!=1) & (train_data.columns[i] not in ['type', 'y']):
            special_var.append(train_data.columns[i])
    print(special_var)


    # pred_date = np.unique(test_data['time'].values)
    pred_date = test_data[test_data['type'] == test_data['type'].values[0]]['time'].values

    # df_final = test_data[~test_data[['time', '']].apply(frozenset, axis=1).duplicated()]
    # print(df_final.shape)

    types = np.unique(test_data['type'].values)
    
    # flag = test_data['type'].values[0]
    # types = [flag]
    # for type in test_data['type'].values:
    #     if type!=flag:
    #         types.append(type)
    #         flag = type

    # pred_dataset, model_date = DataLoading(test_data)
    # TOTALDEVICENUM = pred_dataset.shape[1]-1
    
    TOTALDEVICENUM = len(np.unique(train_data['type'].values))
    FinalResults_EachDevIdx_EachDaySeqIdx = [  [ [ ] for __ in range( args.seq_out_len ) ] for _ in range( TOTALDEVICENUM ) ]
    predictions_EachDeviceEachDay = [[  ] for _ in range(TOTALDEVICENUM) ]
    actuals_EachDeviceEachDay = [[ ] for _ in range(TOTALDEVICENUM) ]
    
 
    """
    Train model and get predictions
    """
    if trained == False:
        dataloader = DataTemplate(train_data, test_data, args.batch_size, args.batch_size, args.batch_size, trained)
        mae, rmse, mape, predictions_EachDeviceEachDay, actuals_EachDeviceEachDay, A = RunGNN(dataloader, loss, trained, TOTALDEVICENUM, args.comp, save_name)
    
    """
    Load pre-trained model and get predictions
    """
    if trained == True:
        dataloader = DataTemplate(train_data, test_data, args.batch_size, args.batch_size, args.batch_size, trained)
        mae, rmse, mape, predictions_EachDeviceEachDay, actuals_EachDeviceEachDay, A = Prediction(dataloader, loss, trained, TOTALDEVICENUM, args.comp, save_name)
    
    """
    Save results
    """

    #File #1: Adjacency Matrix
    for i in range(TOTALDEVICENUM):
        for j in range(TOTALDEVICENUM):
            A[i,j] = A[i,j].item()
    A = pd.DataFrame(A)
    A.to_csv( "Adjacency Matrix.csv", encoding="utf-8-sig", index=False)
   
    #File #2: Predictions 
    tmpTotalOutSampleDay = len(predictions_EachDeviceEachDay[0])

    Results_Prediction = [ [ 0.0 for _ in range( tmpTotalOutSampleDay ) ] for _ in range( TOTALDEVICENUM ) ]
    Results_Actual = [ [ 0.0 for _ in range( tmpTotalOutSampleDay ) ] for _ in range( TOTALDEVICENUM ) ]

    for devIdx in range( TOTALDEVICENUM):
        for dayIdx in range( tmpTotalOutSampleDay ):

            Results_Prediction[devIdx][dayIdx] = np.mean(predictions_EachDeviceEachDay[devIdx][dayIdx])
            Results_Actual[devIdx][dayIdx] = np.mean(actuals_EachDeviceEachDay[devIdx][dayIdx])
           
    # ColumnName_DevIdx = [ "DevIdx_"+str(i+1) for i in range(TOTALDEVICENUM) ]
    ColumnName_DevIdx = list(types)

    Results3 = pd.DataFrame( Results_Prediction )
    Results3 = Results3.transpose()

    Results3.index = pred_date
    Results3.columns = ColumnName_DevIdx

    for var in special_var:
        print(test_data['type'].values[0])
        print(test_data[test_data['type'] == test_data['type'].values[0]])
        Results3[var] = test_data[test_data['type'] == test_data['type'].values[0]][var].values
    Results3.to_csv( "Predictions.csv", encoding="utf-8-sig", index=True)

    #File #3: Evaluation metrics of model for forecasting printer usage of each device

    Final_MAE = [ 0.0 for _ in range( TOTALDEVICENUM ) ]
    Final_MAPE = [ 0.0 for _ in range( TOTALDEVICENUM ) ]
    Final_RMSE = [ 0.0 for _ in range( TOTALDEVICENUM ) ]

    for devIdx in range( TOTALDEVICENUM ):

        tmpPred = torch.from_numpy(np.array(Results_Prediction[devIdx]))
        tmpAct = torch.from_numpy(np.array(Results_Actual[devIdx]))
        metrics = metric(tmpPred, tmpAct)
        Final_MAE[devIdx] = metrics[0]
        Final_RMSE[devIdx] = metrics[1]
        Final_MAPE[devIdx] = metrics[2]
        
    FinalPerformance = [ [ 0 for _ in range( 4 ) ] for _ in range( TOTALDEVICENUM ) ]
  
    for devIdx in range( TOTALDEVICENUM ):
  
        FinalPerformance[ devIdx ][ 0 ] = devIdx
        FinalPerformance[ devIdx ][ 1 ] = Final_MAE[ devIdx ]
        FinalPerformance[ devIdx ][ 2 ] = Final_RMSE[ devIdx ]
        FinalPerformance[ devIdx ][ 3 ] = Final_MAPE[ devIdx ]

    FinalPerformance = pd.DataFrame( FinalPerformance )
    FinalPerformance.columns = [ "DevIdx" , "MAE" , "RootMSE" , "MAPE"]

    FinalPerformance.to_csv( "Evaluation Metrics for each Device.csv", encoding="utf-8-sig", index=False)


    return Results3, FinalPerformance, A

