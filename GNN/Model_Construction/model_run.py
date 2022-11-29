import numpy as np
import torch
torch.manual_seed(42) # Setting the seed
from GNN.Model_Construction.GNN import Trainer
from GNN.Model_Construction.GNN import gtnet
import time

from GNN.Model_Construction.GNN import *
from GNN.Data_Preparation.dataloading import *
from sklearn.model_selection import train_test_split
from GNN.Setting.setting import *
from GNN.Data_Preparation.dataloading import *

args = set()
val_ratio = args.val_ratio

"""
Train, validate and test Graphical Neural Network model
"""
def RunGNN(dataloader, loss_fn, pre, num_nodes, comp, save_name):

    device = torch.device(args.device)

    predefined_A = 0
    input_size = dataloader['x_train'].shape[3]-1

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=input_size, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True, comp = comp)

    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    # print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, device, args.cl, loss_fn, comp)

    # print("start training...",flush=True)
    his_loss =[]
    his_loss_rmse = []
    val_time = []
    train_time = []
    minl = 1e5

    if pre == False:
        for i in range(1,args.epochs+1):

            #train
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()

            # dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                x = x[:,:,:,1:] 
                
                y = y[:,:,:,:] 
                # trainx = torch.Tensor(x).to(device)
                trainx = torch.Tensor(x)
                if comp == 'gpu':
                    trainx = trainx.to(device)
                trainx= trainx.transpose(1, 3)
                # trainy = torch.Tensor(y).to(device)
                trainy = torch.Tensor(y)
                if comp=='gpu':
                    trainy = trainy.to(device)
                trainy = trainy.transpose(1, 3)
                if iter%args.step_size2==0: 
                    perm = np.random.permutation(range(num_nodes))
                num_sub = int(num_nodes/args.num_split) 
                for j in range(args.num_split): 
                    if j != args.num_split-1:
                        id = perm[j * num_sub:(j + 1) * num_sub]
                    else:
                        id = perm[j * num_sub:] 
                  
                    if comp=='gpu':
                        id = torch.tensor(id).to(device).type(torch.long)
                    else:
                        id = torch.tensor(id).type(torch.long)
                    tx = trainx[:, :, id, :]
                    ty = trainy[:, :, id, :] 
                    loss, rmse, mape, A1 = engine.train(tx, ty[:,0,:,:],id)

                    train_loss.append(loss)
                    train_mape.append(mape)
                    train_rmse.append(rmse)
                # if iter % args.print_every == 0 :
                #     log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                #     print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
            t2 = time.time()
            train_time.append(t2-t1)

            #validation
            valid_loss = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                x = x[:,:,:,1:] 
                y = y[:,:,:,:] 

                testx = torch.Tensor(x)
                # testx = torch.Tensor(x).to(device)
                if comp=='gpu':
                    testx = testx.to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y)
                # testy = torch.Tensor(y).to(device)
                if comp=='gpu':
                    testy = testy.to(device)
                testy = testy.transpose(1, 3)
                # print('testx:',testx.shape)
  
                loss, rmse, mape, A = engine.eval(testx, testy[:,0,:,:])
                valid_loss.append(loss)
                valid_mape.append(mape)
                valid_rmse.append(rmse)

            s2 = time.time()
            val_time.append(s2-s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)
            his_loss_rmse.append(mvalid_rmse)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
            
            if mvalid_loss<minl:
                torch.save(engine.model.state_dict(), args.save + save_name +".pth")
                

    engine.model.load_state_dict(torch.load(args.save + save_name + ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_train'])
    realy = realy.transpose(1, 3)[:, 0, :, :]
    for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        x = x[:,:,:,1:] 
        # testx = torch.Tensor(x).to(device)
        testx = torch.Tensor(x)
        if comp=='gpu':
            testx = testx.to(device)
        testx = testx.transpose(1, 3) 
        with torch.no_grad():
            preds, A3 = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    realy2 = torch.Tensor(dataloader['y_val'])
    realy2 = realy2.transpose(1, 3)[:, 0, :, :]
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        x = x[:,:,:,1:] 
        # testx = torch.Tensor(x).to(device)
        testx = torch.Tensor(x)
        if comp=='gpu':
            testx = testx.to(device)
        testx = testx.transpose(1, 3) 
        with torch.no_grad():
            preds, A3 = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat.view(realy.size(0)+realy2.size(0),realy.size(1),realy.size(2))
    # ptrain = scaler.inverse_transform(yhat)[:,:,0]
    ptrain = yhat[:,:,0]
    
    # print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    # print("The valid RMSE on best model is", str(round(his_loss_rmse[bestid],4)))

    #valid data
    outputs = []
    # realy = torch.Tensor(dataloader['y_val']).to(device)
    realy = torch.Tensor(dataloader['y_val'])
    if comp=='gpu':
        realy = realy.to(device)
    realy = realy.transpose(1,3)[:,0,:,:]
    
    for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        x = x[:,:,:,1:] 
        y = y[:,:,:,:] 
        # testx = torch.Tensor(x).to(device)
        testx = torch.Tensor(x)
        if comp=='gpu':
            testx = testx.to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds, A2 = engine.model(testx)
            preds = preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat.view(realy.size(0),realy.size(1),realy.size(2))

    # pred = scaler.inverse_transform(yhat)
    pred = yhat
    vmae, vrmse, vmape= metric(pred,realy)

    #test data
    outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = torch.Tensor(dataloader['y_test'])
    if comp=='gpu':
        realy = realy.to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]


    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x = x[:,:,:,1:] 
        # testx = torch.Tensor(x).to(device)
        testx = torch.Tensor(x)
        if comp=='gpu':
            testx = testx.to(device)
        testx = testx.transpose(1, 3) 
        with torch.no_grad():
            preds, A3 = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat.view(realy.size(0),realy.size(1),realy.size(2))

    mae = []
    mape = []
    rmse = []


    predictions_EachDeviceEachDay = [[ [ ] for __ in range( dataloader['x_test'].shape[ 0 ] + args.seq_out_len - 1 ) ] for _ in range(num_nodes) ]
    actuals_EachDeviceEachDay = [[ [ ] for __ in range( dataloader['x_test'].shape[ 0 ] + args.seq_out_len - 1 ) ] for _ in range(num_nodes) ]

    for i in range(args.seq_out_len):
        # pred = scaler.inverse_transform(yhat[:, :, i])
        pred = yhat[:, :, i]
        real = realy[:, :, i]

        for batchIdx in range(realy.size(0)):
            for devIdx in range(realy.size(1)):
                predictions_EachDeviceEachDay[devIdx][batchIdx + i].append(pred[batchIdx,devIdx].cpu().item())
                actuals_EachDeviceEachDay[devIdx][batchIdx + i].append(real[batchIdx,devIdx].cpu().item())


        metrics = metric(pred, real)
        mae.append(metrics[0])
        rmse.append(metrics[1])
        mape.append(metrics[2])


    return mae, rmse, mape, predictions_EachDeviceEachDay, actuals_EachDeviceEachDay, A3


def Prediction(dataloader, loss_fn, pre, num_nodes, comp, save_name):

    device = torch.device(args.device)

    predefined_A = 0
    input_size = dataloader['x_test'].shape[3]-1

    model = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, num_nodes,
                  device, predefined_A=predefined_A,
                  dropout=args.dropout, subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, end_channels= args.end_channels,
                  seq_length=args.seq_in_len, in_dim=input_size, out_dim=args.seq_out_len,
                  layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=True, comp = comp)

    # print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    # print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, device, args.cl, loss_fn, comp)

    engine.model.load_state_dict(torch.load(args.save + save_name + ".pth"))
    device = torch.device(args.device)
    #test data
    outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = torch.Tensor(dataloader['y_test'])
    if comp=='gpu':
        realy = realy.to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]


    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        x = x[:,:,:,1:] 
        # testx = torch.Tensor(x).to(device)
        testx = torch.Tensor(x)
        if comp=='gpu':
            testx = testx.to(device)
        testx = testx.transpose(1, 3) 
        with torch.no_grad():
            preds, A3 = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat.view(realy.size(0),realy.size(1),realy.size(2))

    mae = []
    mape = []
    rmse = []


    predictions_EachDeviceEachDay = [[ [ ] for __ in range( dataloader['x_test'].shape[ 0 ] + args.seq_out_len - 1 ) ] for _ in range(num_nodes) ]
    actuals_EachDeviceEachDay = [[ [ ] for __ in range( dataloader['x_test'].shape[ 0 ] + args.seq_out_len - 1 ) ] for _ in range(num_nodes) ]

    for i in range(args.seq_out_len):
        # pred = scaler.inverse_transform(yhat[:, :, i])
        pred = yhat[:, :, i]
        real = realy[:, :, i]

        for batchIdx in range(realy.size(0)):
            for devIdx in range(realy.size(1)):
                predictions_EachDeviceEachDay[devIdx][batchIdx + i].append(pred[batchIdx,devIdx].cpu().item())
                actuals_EachDeviceEachDay[devIdx][batchIdx + i].append(real[batchIdx,devIdx].cpu().item())


        metrics = metric(pred, real)
        mae.append(metrics[0])
        rmse.append(metrics[1])
        mape.append(metrics[2])


    return mae, rmse, mape, predictions_EachDeviceEachDay, actuals_EachDeviceEachDay, A3


