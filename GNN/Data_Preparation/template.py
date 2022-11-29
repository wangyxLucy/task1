from GNN.Data_Preparation.dataloading import *
from GNN.Data_Preparation.feature import *

import numpy as np
from GNN.Setting.setting import *

args = set()


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


"""
Generate GNN-based data
"""
def DataTemplate(train, test, batch_size, valid_batch_size= None, test_batch_size=None, trained=False):
    
    dataloader = {}
    day = args.seq_out_len


    TOTALDEVICENUM = len(np.unique(test['type'].values))
    types = np.unique(test['type'].values)

    x_offsets = np.sort(
    np.concatenate((np.arange(-day+1, 1, 1),))
    )
    y_offsets = np.sort(np.arange(-day+1, 1, 1))
        

    if trained == False:
        ###train & validation
        # df, model_date = DataLoading(train)
        # print('df:', df)
    

        # totalTrainSize = df.shape[0]- args.seq_out_len + 1
        # TOTALDEVICENUM = df.shape[1]-1
       

        # df = df.iloc[:,0:TOTALDEVICENUM+1] 

        # features = FeatureGen(0,df)
        
        # features = features.drop('target', axis = 1)  
        # print('features:', features.shape)
        # df = df.set_index(['Datetime'])

        # data = np.expand_dims(df.values, axis=-1)

        # data_list = [data]

        # day_of_week = np.tile(features, [TOTALDEVICENUM, 1 , 1]).transpose((1, 0 , 2))
        # print('day_of_week:', day_of_week.shape)
        
        # data_list.append(day_of_week)

        # #-> Get the data list combined with time_in_day
        # data = np.concatenate(data_list, axis=-1)
        # print('data:', data.shape) #data: (78, 37, 63)

        #########!!!!!!!!###########

        data_list = []
        for i in range(TOTALDEVICENUM):
            train_data = train[train['type'] == types[i]]
            features = FeatureGen(train_data).values
            data_list.append(features)
        data = np.array(data_list)
        # data = np.concatenate(data_list, axis=1)
        data = data.reshape((data.shape[1], data.shape[0], data.shape[2]))

        x, y = [], []
        # t is the index of the last observation.
        totalTrainSize = features.shape[0] - args.seq_out_len + 1
        min_t = abs(min(x_offsets))
        max_t = abs(totalTrainSize + args.seq_out_len -1 - abs(max(y_offsets)))  

        x, y = [], [] 
        for t in range(min_t, max_t): 

            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
            
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        
        num_train = int((875/993) * features.shape[0])


        x_train, y_train = x[:num_train], y[:num_train]
        
        x_val, y_val = (
            x[num_train: ],
            y[num_train: ],
        )

        dataloader['x_train'] = x_train
        dataloader['y_train'] = y_train
        dataloader['x_val'] = x_val
        dataloader['y_val'] = y_val
        
        scaler = StandardScaler(mean=dataloader['x_train'][..., 0].mean(), std=dataloader['x_train'][..., 0].std())

        # Data format

        # for category in ['train', 'val']:
        #     dataloader['x_' + category][..., 0] = scaler.transform(dataloader['x_' + category][..., 0])

        dataloader['train_loader'] = DataLoaderM(dataloader['x_train'], dataloader['y_train'], batch_size)
        dataloader['val_loader'] = DataLoaderM(dataloader['x_val'], dataloader['y_val'], valid_batch_size)

        dataloader['scaler'] = scaler
    
    ###test###
    
    # test_df, pred_date = DataLoading(test)
    # test_features = FeatureGen(0,test_df)
    # TOTALDEVICENUM = test_df.shape[1]-1

    # totalTestSize = test_df.shape[0]- args.seq_out_len + 1
  
    # test_df = test_df.iloc[:,0:TOTALDEVICENUM+1] 

    # features = test_features
    # features = features.drop('target', axis = 1)  

    # df = test_df.set_index(['Datetime'])

    # x_offsets = np.sort(
    # np.concatenate((np.arange(-day+1, 1, 1),))
    # )
    # y_offsets = np.sort(np.arange(-day+1, 1, 1))
    
    # data = np.expand_dims(df.values, axis=-1)
    # data_list = [data]

    # day_of_week = np.tile(features, [TOTALDEVICENUM, 1 , 1]).transpose((1, 0 , 2))
    
    # data_list.append(day_of_week)


    #########!!!!!!!!###########
    data_list = []
    for i in range(TOTALDEVICENUM):
        test_data = test[test['type'] == types[i]]
        features = FeatureGen(test_data).values
        data_list.append(features)

    data = np.array(data_list)
    data = data.reshape((data.shape[1], data.shape[0], data.shape[2]))
    x2, y2 = [], []
    # t is the index of the last observation.
    totalTestSize = features.shape[0] - args.seq_out_len + 1
    min_t = abs(min(x_offsets)) 
    max_t = abs(totalTestSize + args.seq_out_len -1 - abs(max(y_offsets)))  

    for t in range(min_t, max_t): 

        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x2.append(x_t)
        y2.append(y_t)
        
    x_test = np.stack(x2, axis=0)
    y_test = np.stack(y2, axis=0)
    
    dataloader['x_test'] = x_test
    dataloader['y_test'] = y_test

    # dataloader['x_test'][..., 0] = scaler.transform(dataloader['x_test'][..., 0])

    dataloader['test_loader'] = DataLoaderM(dataloader['x_test'], dataloader['y_test'], test_batch_size)
    # dataloader['scaler'] = scaler

    return dataloader

