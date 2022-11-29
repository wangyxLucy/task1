
from GNN.Test.test import *
from GNN.Setting.setting import *
from datetime import datetime
from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings("ignore")

#Set parameters, such as: epochs
args = set()

#Read data
# train_data = pd.read_csv('Data/HK_demand_modeling_input_level_2train.csv')
# test_data = pd.read_csv('Data/HK_demand_modeling_input_level_2test.csv')
train_data = pd.read_csv('Data/new_train.csv')
test_data = pd.read_csv('Data/new_test.csv')

save_name = 'test'


### 1 ###
#Call the GNN forecaster, train the model and get predictions
Predictions, Metrics, Adj = Test(train_data, test_data, 0, False, save_name) 

### 2 ###
#Call a trained model saved and get predictions

#Case 1: If all features are generated based on time, that is, the columns of data only include ['type', 'time', 'y']
# type_list = train_data['type'].unique()
# time_list = [datetime.strftime(datetime(2022, 5, 18) + relativedelta(days=d), "%Y/%m/%d") for d in range(20)]
# type_df = pd.DataFrame(type_list, columns=['type'])
# type_df['tmp'] = 0
# time_df = pd.DataFrame(time_list, columns=['time'])
# time_df['tmp'] = 0
 
# test_data = type_df.merge(time_df, on='tmp').drop('tmp', 1)
# test_data['y'] = 0

# train_data = 0 

#Case 2: If other features are included. Read data from the files
Predictions, Metrics, Adj = Test(train_data, test_data, 0, True, save_name) 






    