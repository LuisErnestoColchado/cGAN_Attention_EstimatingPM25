# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: cGANSL Model (Adversarial and Spatial Loss)
# ******************************************************************************************
import os, sys
from Experimentation.attention_NN import DATASOURCE

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from commom import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
import pandas as pd 


class cGAN:
    def __init__(self, generator, discriminator, z_dim, g_steps, d_steps, num_epoch, scale_pollutant, scale_distance, 
                 g_optimizer, d_optimizer, parameter_spatial, parameter_adversarial, spatial_loss=True, cuda=True, train_labels=11):#, size_batch=128):
        self.generator = generator
        self.discriminator  = discriminator
        self.criterion_adv = torch.nn.BCELoss()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.num_epoch = num_epoch
        self.z_dim = z_dim
        self.cuda = cuda
        self.g_steps = g_steps
        self.d_steps = d_steps
        self.train_labels = train_labels

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.spatial_loss = spatial_loss
        if spatial_loss:
            self.parameter_spatial = parameter_spatial
            self.parameter_adversarial = parameter_adversarial  # 1
        else:
            self.parameter_spatial = 0
            self.parameter_adversarial = 1  
        self.scale_pollutant = scale_pollutant
        self.scale_distance = scale_distance

    def training_test(self, train_loader, condition_test, x_test, knn, directory_results, station):
        
        run_directory = f'{directory_results}/cgan_runs'
        if not os.path.isdir(run_directory):
            os.mkdir(run_directory)

        writer = SummaryWriter(log_dir=run_directory)       

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.criterion_adv.cuda()
            
        self.generator = init_net(self.generator)
        self.discriminator = init_net(self.discriminator)

        if not os.path.isdir('../Models/cGANSL/Backup'):
            os.mkdir('../Models/cGANSL/Backup')
        
        init_g = f"../Models/cGANSL/Backup/g_model0_{DATASOURCE}_{knn}.pt"
        init_d = f"../Models/cGANSL/Backup/d_model0_{DATASOURCE}_{knn}.pt"
        
        try:
            self.generator.load_state_dict(torch.load(init_g))
        except Exception as e:
            print(e, ' Creating new backup of initial Generator ...')
            torch.save(self.generator.state_dict(), init_g)
        
        try:
            self.generator.load_state_dict(torch.load(init_d))
        except Exception as e:
            print(e, ' Creating new backup of initial Discriminator ...')
            torch.save(self.generator.state_dict(), init_d)

        real_error_discriminator = []
        fake_error_discriminator = []
        fake_error_generator = []
        fake_mae_test = []
        fake_rmse_test = []
        fake_mse_test = []
        fake_r2_test = []
        fake_r2_test = []

        backup_directory = f'../Models/cGANSL/Backup/cgansl_advloss{self.parameter_adversarial}_spl{self.parameter_spatial}_knn{knn}'
        if not os.path.isdir(backup_directory):
            os.mkdir(backup_directory)

        with tqdm(total=self.num_epoch) as bar:
            for i in range(1, self.num_epoch+1):
                for condition, distances, knn_values, x_real in train_loader:
                    condition = Variable(condition, requires_grad=False)
                    x_real = Variable(x_real, requires_grad=False)

                    for _ in range(self.g_steps):
                        size_batch = condition.size(0)
                        
                        indeces_labeled = torch.where(~torch.isnan(x_real))

                        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
                        valid = Variable(FloatTensor(len(indeces_labeled[0])).fill_(1.0), requires_grad=False)
                        fake = Variable(FloatTensor(len(indeces_labeled[0])).fill_(0.0), requires_grad=False)
    
                        z = FloatTensor(np.random.rand(size_batch, self.z_dim))#0, 1, 
                        self.generator.zero_grad()
 
                        fake_input_generator = torch.cat([z, condition ], -1)
                        fake_data = self.generator(fake_input_generator)
                        distances_inv = -1 * distances

                        # Softmax to spatial distances
                        softmax_dist = torch.nn.Softmax(dim=1)(distances_inv)
                        
                        spatial_error = torch.mul(torch.abs(knn_values - fake_data), softmax_dist) 
                        
                        spatial_error_ = spatial_error.mean()
                        fake_data_labeled = fake_data[indeces_labeled]
                        condition_labeled = condition[indeces_labeled]

                        fake_input_discriminator = torch.cat((fake_data_labeled, condition_labeled), dim=1)
                        dg_fake_decision = self.discriminator(fake_input_discriminator)
                        
                        adv_error = self.parameter_adversarial * self.criterion_adv(dg_fake_decision.reshape(len(dg_fake_decision)), valid)
                        spatial_error_ = self.parameter_spatial * spatial_error_
                        
                        g_error = adv_error + spatial_error_
                        
                                   
                        g_error.backward(retain_graph=True)
                        self.g_optimizer.step()
                        ge = extract(g_error)[0]

                    for _ in range(self.d_steps):
                        x_real_labeled = x_real[indeces_labeled].reshape(len(x_real[indeces_labeled]))
                        self.discriminator.zero_grad()
                        real_input_discriminator = torch.cat((x_real_labeled.reshape(len(x_real_labeled), 1), condition_labeled), dim=1)

                        real_decision = self.discriminator(real_input_discriminator)
                        
                        d_real_error = self.criterion_adv(real_decision.reshape(len(real_decision)), valid)

                        d_fake_decision = self.discriminator(fake_input_discriminator.detach())

                        d_fake_error = self.criterion_adv(d_fake_decision.reshape(len(d_fake_decision)), fake)
                        #!print(d_real_error, d_fake_error)
                        d_loss = d_real_error + d_fake_error
                        d_loss.backward()
                        self.d_optimizer.step()
                        dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
                with torch.no_grad():
                    
                    size_test = condition_test.size(0)

                    z_test = FloatTensor(np.random.rand(size_test, self.z_dim))

                    test_input_generator = torch.cat((z_test, condition_test), -1)
                
                    test_output_generator = self.generator(test_input_generator)

                    real_pm25 = self.scale_pollutant.inverse_transform(x_test.reshape(-1, 1).to('cpu').detach().numpy())
                    fake_pm25 = self.scale_pollutant.inverse_transform(test_output_generator.reshape(-1, 1).to('cpu').detach().numpy())

                    MSE_test = mean_squared_error(real_pm25, fake_pm25)
                    RMSE_test = round(np.sqrt(MSE_test), 4)
                    MAE_test = round(mean_absolute_error(real_pm25, fake_pm25), 4)
                    r2_test = round(r2_score(real_pm25, fake_pm25), 4)

                    fake_mse_test.append(MSE_test)
                    fake_rmse_test.append(RMSE_test)
                    fake_mae_test.append(MAE_test)
                    fake_r2_test.append(r2_test)

                    fake_error_generator.append(ge)
                    real_error_discriminator.append(dre)
                    fake_error_discriminator.append(dfe)

                    bar.set_description("Epoch " + str(i))
                    bar.update(1)

                    torch.save(self.generator.state_dict(), f'{backup_directory}/g_model_{DATASOURCE}_{i}_{station}.pt')
                    torch.save(self.discriminator.state_dict(), f'{backup_directory}/d_model_{DATASOURCE}_{i}_{station}.pt')
                    writer.add_scalars('Fake Loss', {'Fake Generator': ge, 'Loss Fake Discriminator': dfe}, i)
                    writer.add_scalar('Testing RMSE', RMSE_test, i)
                    writer.add_scalar('Testing MAE', MAE_test, i)
                    writer.add_scalar('Testing R2', r2_test, i) 

                    msg = f'''Test results epoch {i} -> MSE: {MSE_test}, RMSE: {RMSE_test}, MAE: {MAE_test}, R2: {r2_test},
                        , GE: {ge}, DFE: {dfe}, Real dist: {stats(extract(x_real_labeled))}, Fake dist: {stats(extract(fake_data_labeled))} 
                        '''
                    print(msg)
                    
            results = {"mse_test": fake_mse_test, "rmse_test": fake_rmse_test, "mae_test": fake_mae_test, 
            "r2_test": fake_r2_test, "fake_error_g": fake_error_generator, "fake_error_d": fake_error_discriminator,
            "real_error_d": real_error_discriminator, "output_test": fake_pm25}
            return results, msg