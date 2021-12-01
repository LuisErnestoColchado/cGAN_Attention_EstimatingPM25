import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from commom import *
from tqdm import tqdm


class cGAN:
    def __init__(self, generator, discriminator, z_dim, g_steps, d_steps, num_epoch, scale_pollutant, scale_distance, 
                 g_optimizer, d_optimizer, spatial_loss=True, parameter_spatial=0.5, cuda=True):#, size_batch=128):
        self.generator = generator
        self.discriminitor  = discriminator
        self.criterion_adv = torch.nn.BCELoss()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.num_epoch = num_epoch
        self.z_dim = z_dim
        self.cuda = cuda
        self.g_steps = g_steps
        self.d_steps = d_steps

        #if g_optimizer == 'Adam':
        self.g_optimizer = g_optimizer
        #if d_optimizer == 'Adam':
        self.d_optimizer = d_optimizer

        if spatial_loss:
            self.parameter_spatial = parameter_spatial
            self.parameter_adversarial = 1  # 1
        else:
            self.parameter_spatial = 0
            self.parameter_adversarial = 1
        
        self.scale_pollutant = scale_pollutant
        scale_distance = scale_distance

    
    def training_test(self, train_loader, x_test, y_test):

        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.criterion_adv.cuda()
            
        self.generator = init_net(self.generator)
        self.discriminator = self.init_net(self.discriminator)

        torch.save(self.generator.state_dict(), 'g_model0.pt')
        torch.save(self.discriminator.state_dict(), 'd_model0.pt')

        real_error_discriminator = []
        fake_error_discriminator = []
        fake_error_generator = []
        fake_mae_test = []
        fake_rmse_test = []
        fake_mse_test = []
        fake_r2_test = []

        with tqdm(total=self.num_epoch) as bar:
            for i in range(1, self.num_epoch+1):
                for x_batch, distances, neighbors, y_batch in train_loader:
                    for _ in range(self.g_steps):
                        # print(dist_batch)
                        # print(type(x_batch), type(dist_batch))
                        size_batch = x_batch.size(0)

                        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
                        
                        valid = FloatTensor(size_batch, 1).fill_(1.0)
                        fake = FloatTensor(size_batch, 1).fill_(0.0) 

                        z = FloatTensor(np.random.normal(0, 1, (size_batch, self.z_dim)))
                        #valid = FloatTensor(size_batch, 1).fill_(1.0)
                        #fake = FloatTensor(size_batch, 1).fill_(0.0)

                        # = dist_batch[:, :10]
                        # = dist_batch[:, 10:]

                        #z = FloatTensor(np.random.normal(0, 1, (size_batch, z_dim)))

                        self.generator.zero_grad()

                        fake_input_generator = torch.cat([z, x_batch], -1)

                        fake_data = self.generator(fake_input_generator)

                        spatial_error = torch.mul(torch.pow(neighbors - fake_data, 2), torch.exp(-distances))

                        spatial_error_ = spatial_error.mean()

                        fake_input_discriminator = torch.cat((fake_data, x_batch), -1)

                        dg_fake_decision = self.discriminator(fake_input_discriminator)

                        g_error = (self.parameter_adversarial * self.criterion_adv(dg_fake_decision, self.valid)) + (
                                    self.parameter_spatial * spatial_error_)

                        g_error.backward(retain_graph=True)
                        self.g_optimizer.step()
                        ge = extract(g_error)[0]

                for _ in range(self.d_steps):
                    self.discriminator.zero_grad()

                    real_input_discriminator = torch.cat((y_batch.view((len(y_batch), 1)), x_batch), -1)

                    real_decision = self.discriminator(real_input_discriminator)

                    d_real_error = self.criterion_valid(real_decision, valid)

                    d_fake_decision = self.discriminator(fake_input_discriminator)

                    d_fake_error = self.criterion_valid(d_fake_decision, fake)

                    d_loss = (d_real_error + d_fake_error)
                    d_loss = Variable(d_loss.detach_(), requires_grad=True)
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()
                    dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
            
                    with torch.no_grad():

                        size_test = x_test.size(0)

                        z_test = FloatTensor(np.random.normal(0, 1, (size_test, z_dim)))

                        test_input_generator = torch.cat((z_test, x_test), -1)

                        test_output_generator = self.generator(test_input_generator)

                        real_pm25 = self.scale_pollutant.inverse_transform(y_test.reshape(-1, 1).to('cpu').detach().numpy())
                        fake_pm25 = self.scale_pollutant.inverse_transform(
                            test_output_generator.reshape(-1, 1).to('cpu').detach().numpy())

                        MSE_test = mean_squared_error(real_pm25, fake_pm25)
                        RMSE_test = np.sqrt(MSE_test)
                        MAE_test = mean_absolute_error(real_pm25, fake_pm25)
                        r2_test = r2_score(real_pm25, fake_pm25)

                        fake_mse_test.append(MSE_test)
                        fake_rmse_test.append(RMSE_test)
                        fake_mae_test.append(MAE_test)
                        fake_r2_test.append(r2_test)

                        fake_error_generator.append(ge)
                        real_error_discriminator.append(dre)
                        fake_error_discriminator.append(dfe)

                        bar.set_description("Epoch " + str(i))
                        bar.update(1)
                        print(" Test metrics: MSE ", MSE_test, " RMSE ", RMSE_test, " MAE ", MAE_test, " R2 ", r2_test, 
                                " GE ", ge,  " DFE ", dfe)  # )