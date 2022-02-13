# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Attention model
# ******************************************************************************************
from calendar import EPOCH
from datetime import datetime
import os, sys

from soupsieve import match
currentdir = os.path.dirname(os.path.realpath(__file__) )
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from Models.commom import * 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Attention_model:
    def __init__(self, attention_layer, ann, knn, criterion, opt_atten_layer, opt_ann, num_epoch,
                 scale_pollutant):
        self.attention_layer = attention_layer
        self.ann = ann
        self.knn = knn
        self.criterion = criterion
        self.opt_atten_layer = opt_atten_layer
        self.opt_ann = opt_ann
        self.num_epoch = num_epoch 
        self.scale_pollutant = scale_pollutant

        self.writer = SummaryWriter(log_dir=f'attention_runs/{str(datetime.now())}')

    def training(self, data_loader, data_loader_test):
        R2_testing = []
        RMSE_testing = []
        MAE_testing = []
        rmse_test = np.zeros((len(data_loader_test[0]['x_test'])))
        mae_test = np.zeros((len(data_loader_test[0]['x_test'])))
        r2_test = np.zeros((len(data_loader_test[0]['x_test'])))
        with tqdm(total=self.num_epoch) as bar:
            for i in range(1, self.num_epoch+1):
                Y = np.zeros((len(data_loader), 1))
                Y_HAT = np.zeros((len(data_loader), 1))
                #for j in range(1, int(len(x)/10.0)+1):
                for j, batch_data in enumerate(data_loader):
                    #print(type(batch_data))
                    x_batch = batch_data['x_train']
                    y_batch = batch_data['y_train']
                    #x_batch = torch.cat([features, current_test], dim=0)
                    #!print(x_batch.shape)
                    out, result, a = self.attention_layer(x_batch)

                    out_final = self.ann(result)
                    
                    error = self.criterion(y_batch, out_final)
                    
                    self.opt_atten_layer.zero_grad()
                    self.opt_ann.zero_grad()
                    error.backward()
                    self.opt_atten_layer.step()
                    self.opt_ann.step()
                with torch.no_grad():
                    
                    y_real = np.zeros((len(data_loader_test[0]['x_test']), len(data_loader_test)))
                    y_hat = np.zeros((len(data_loader_test[0]['x_test']), len(data_loader_test)))
                    for cd, batch_data_test in enumerate(data_loader_test):
                        x_batch_test = batch_data_test['x_test']
                        y_batch_test = batch_data_test['y_test']
                        #x_batch_test = x_test[j, :]
                        #y_batch_test = y_test[j]
                        
                        #!graph_test = torch.cat([x_batch, x_batch_test], dim=0)
                        #print(graph_test.shape)
                        #x_ = x_batch#torch.cat([x_batch, x_t[j-1, :].reshape(1, len(x_t[j-1, :]))], dim=0)
                        #y_ = y_t[j-1, :]

                        _, z, a_test = self.attention_layer(x_batch_test)

                        y_hat_test = self.ann(z)

                        #y_hat_test = y_hat
                        
                        y_hat_test = self.scale_pollutant.inverse_transform(y_hat_test.detach().numpy().reshape(-1, 1))
                        y = self.scale_pollutant.inverse_transform(y_batch_test.detach().numpy().reshape(-1, 1))
                        
                        #print(EPOCH)
                        #print('y_hat ', y_hat_test)
                        #print('*******************************************')
                        #print('y ', y)
                        #!Y[bt] = y
                        #!Y_HAT[bt] = y_hat_test
                        for ct in range(len(y)):
                            y_real[ct][cd] = y[ct]
                            y_hat[ct][cd] = y_hat_test[ct]
                            #mae_test[ct] += mean_absolute_error(y[ct], y_hat_test[ct])
                            #sprint(Y, Y_HAT)
                            #r2_test[ct] += r2_score(y[ct], y_hat_test[ct])
                for ct in range(len(y)):
                    #print('aca ', y_real[ct])
                    #print('ac 2 ', y_hat[ct])
                    rmse_test[ct] = np.sqrt(mean_squared_error(y_real[ct], y_hat[ct])) 
                    mae_test[ct] = mean_absolute_error(y_real[ct], y_hat[ct])
                    r2_test[ct] = r2_score(y_real[ct], y_hat[ct])

                
                rmse_test_mean = np.mean(rmse_test)
                mae_test_mean = np.mean(mae_test)
                r2_test_mean = np.mean(r2_test)
                        #print('ACA  ', rmse_test[i][ct] )
                MAE_testing.append(mae_test)
                RMSE_testing.append(rmse_test)
                R2_testing.append(r2_test)
                self.writer.add_scalar('Testing RMSE', rmse_test_mean, i)
                self.writer.add_scalar('Testing MAE', mae_test_mean, i)
                self.writer.add_scalar('Testing R2', r2_test_mean, i) 

                #!"rmse_test_mean = np.mean(rmse_test, axis=0)
                #!"mae_test_mean = np.mean(mae_test, axis=0)
                #!"r2_test_mean = np.mean(r2_test, axis=0)
                #rmse_test_mean = np.mean(rmse_test_mean)
                #mae_test_mean = np.mean(mae_test_mean)
                #r2_test_mean = np.mean(r2_test_mean)

                bar.update(1)
                msg = f'Epoch {i}, RMSE {rmse_test_mean}, MAE {mae_test_mean}, R2 {r2_test_mean}    {len(data_loader_test)}'
                print(msg)
        return self.attention_layer, self.ann, {"rmse_test": RMSE_testing, "mae_test": MAE_testing, "r2_test": R2_testing}, msg

##

