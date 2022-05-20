# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Attention model
# ******************************************************************************************
from datetime import datetime
import os, sys

from numpy import DataSource
currentdir = os.path.dirname(os.path.realpath(__file__) )
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from Models.commom import * 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Attention_model:
    def __init__(self, attention_layer, ann, knn, criterion, opt_atten_layer, opt_ann, num_epoch,
                 scale_pollutant, kernel, DATASOURCE):
        self.attention_layer = attention_layer
        self.ann = ann
        self.knn = knn
        self.criterion = criterion
        self.opt_atten_layer = opt_atten_layer
        self.opt_ann = opt_ann
        self.num_epoch = num_epoch 
        self.scale_pollutant = scale_pollutant

        
        self.dir_backup = f"../Models/Attention/Backup/{DATASOURCE}_{kernel}"
        self.dir_init = f"../Models/Attention/Backup/init/{DATASOURCE}"
        self.kernel = kernel
        self.DATASOURCE = DATASOURCE
        
    def training(self, data_loader, dir_result):
        
        run_directory = f'{dir_result}/attention_runs'
        if not os.path.isdir(run_directory):
            os.mkdir(run_directory)

        writer = SummaryWriter(log_dir=run_directory)

        if not os.path.isdir('../Models/Attention/Backup'):
            os.mkdir('../Models/Attention/Backup')
        
        if not os.path.isdir('../Models/Attention/Backup/init'):
            os.mkdir('../Models/Attention/Backup/init')
            
        if not os.path.isdir(self.dir_backup):
            os.mkdir(self.dir_backup)
        
        if not os.path.isdir(self.dir_init):
            os.mkdir(self.dir_init)

        init_ann = f"{self.dir_init}/ann_init_{self.DATASOURCE}.pt"
        #if os.path.exists(init_ann):
        try:
            self.ann.load_state_dict(torch.load(init_ann))
        except Exception as e:
            print(e, ' Creating new backup of initial ANN ...')
            torch.save(self.ann.state_dict(), init_ann)

        R2_testing = []
        RMSE_testing = []
        MAE_testing = []

        with tqdm(total=self.num_epoch) as bar:
            for i in range(1, self.num_epoch+1):
                Y = np.zeros((len(data_loader), 1))
                Y_HAT = np.zeros((len(data_loader), 1))
                for j, batch_data in enumerate(data_loader):
                    x_batch = batch_data['x_train']
                    y_batch = batch_data['y_train']
                    x_batch_test = batch_data['x_test']
                    y_batch_test = batch_data['y_test']
                    
                    _, result, _ = self.attention_layer(x_batch)

                    out_final = self.ann(result, self.DATASOURCE)
                    
                    error = self.criterion(y_batch, out_final)
                    
                    self.opt_atten_layer.zero_grad()
                    self.opt_ann.zero_grad()
                    error.backward()
                    self.opt_atten_layer.step()
                    self.opt_ann.step()
                    with torch.no_grad():
                        graph_test = torch.cat([x_batch, x_batch_test], dim=0)
                        _, z, _ = self.attention_layer(graph_test)

                        y_hat = self.ann(z, self.DATASOURCE)

                        y_hat_test = y_hat[-1, :]
                        
                        y_hat_test = self.scale_pollutant.inverse_transform(y_hat_test.detach().numpy().reshape(-1, 1))
                        y = self.scale_pollutant.inverse_transform(y_batch_test.detach().numpy().reshape(-1, 1))
                        Y[j] = y
                        Y_HAT[j] = y_hat_test
                torch.save(self.attention_layer, f"{self.dir_backup}/attention_layer_{i}.pt")
                torch.save(self.ann, f"{self.dir_backup}/ann_{i}.pt")
                rmse_test = round(np.sqrt(mean_squared_error(Y, Y_HAT)), 4)
                mae_test = round(mean_absolute_error(Y, Y_HAT), 4)
                r2_test = round(r2_score(Y, Y_HAT), 4)
                MAE_testing.append(mae_test)
                RMSE_testing.append(rmse_test)
                R2_testing.append(r2_test)
                writer.add_scalar('Testing RMSE', rmse_test, i)
                writer.add_scalar('Testing MAE', mae_test, i)
                writer.add_scalar('Testing R2', r2_test, i) 
                msg = f'Epoch {i}, RMSE {rmse_test}, MAE {mae_test}, R2 {r2_test}'
                print(msg)
                bar.update(1)
        return self.attention_layer, self.ann, {"rmse_test": RMSE_testing, "mae_test": MAE_testing, "r2_test": R2_testing, "output_test": Y_HAT}, msg

##

