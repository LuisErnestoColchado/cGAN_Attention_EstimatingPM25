# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Attention model
# ******************************************************************************************
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from Models.commom import * 
from tqdm import tqdm 

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

    def training(self, data_loader):
        R2_testing = []
        RMSE_testing = []
        MAE_testing = []

        with tqdm(total=self.num_epoch) as bar:
            for i in range(1, self.num_epoch+1):
                Y = np.zeros((len(data_loader), 1))
                Y_HAT = np.zeros((len(data_loader), 1))
                #for j in range(1, int(len(x)/10.0)+1):
                for j, batch_data in enumerate(data_loader):
                    #print(type(batch_data))
                    x_batch = batch_data['x_train']
                    y_batch = batch_data['y_train']
                    x_batch_test = batch_data['x_test']
                    y_batch_test = batch_data['y_test']
                    #x_batch = torch.cat([features, current_test], dim=0)
                    
                    out, result, a = self.attention_layer(x_batch)

                    out_final = self.ann(result)
                    
                    error = self.criterion(y_batch, out_final)
                    
                    self.opt_atten_layer.zero_grad()
                    self.opt_ann.zero_grad()
                    error.backward()
                    self.opt_atten_layer.step()
                    self.opt_ann.step()
                    with torch.no_grad():
                        #x_batch_test = x_test[j, :]
                        #y_batch_test = y_test[j]
                        
                        graph_test = torch.cat([x_batch, x_batch_test], dim=0)

                        #x_ = x_batch#torch.cat([x_batch, x_t[j-1, :].reshape(1, len(x_t[j-1, :]))], dim=0)
                        #y_ = y_t[j-1, :]

                        _, z, a_test = self.attention_layer(graph_test)

                        y_hat = self.ann(z)

                        y_hat_test = y_hat[-1, :]
                        
                        y_hat_test = self.scale_pollutant.inverse_transform(y_hat_test.detach().numpy().reshape(-1, 1))
                        y = self.scale_pollutant.inverse_transform(y_batch_test.detach().numpy().reshape(-1, 1))
                        Y[j] = y
                        Y_HAT[j] = y_hat_test
                    
                rmse_test = np.sqrt(mean_squared_error(Y, Y_HAT))
                mae_test = mean_absolute_error(Y, Y_HAT)
                #sprint(Y, Y_HAT)
                r2_test = r2_score(Y, Y_HAT)
                MAE_testing.append(mae_test)
                RMSE_testing.append(rmse_test)
                R2_testing.append(r2_test)
                bar.update(1)
                print('Epoch %d, RMSE %f, MAE %f, R2 %f' % (i, rmse_test, mae_test, r2_test))
                #print(a_test)

