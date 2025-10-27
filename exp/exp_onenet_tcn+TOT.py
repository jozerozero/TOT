from data.data_loader_freq import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.ncca import TSEncoder, GlobalLocalMultiscaleTSEncoder, TSEncoderTime
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
import pdb
import numpy as np
from einops import rearrange
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils.buffer import Buffer
import wandb
import torch.nn.functional as F
from functorch import vmap, jacfwd, grad
import os
import time
from pathlib import Path
import torch.distributions as D
import warnings
warnings.filterwarnings('ignore')

class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, mlp_width, mlp_depth, mlp_dropout, act=nn.ReLU()):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.act = act

    def forward(self, x, train=True):
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        x = self.output(x)
        # x = F.sigmoid(x)
        return x

class MLP2(nn.Module):
    """A simple MLP with ReLU activations"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, leaky_relu_slope=0.2):
        super().__init__()
        layers = []
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)

class NPObsLatentTransitionPrior(nn.Module):

    def __init__(
            self,
            lags,
            latent_size,
            num_layers=3,
            hidden_dim=64,
            compress_dim=3):
        super().__init__()
        self.L = 1
        self.latent_size = latent_size
        self.gs = nn.ModuleList([MLP2(input_dim=compress_dim , hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in
                                 range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP2(input_dim=self.L * (latent_size*2) + 1, hidden_dim=hidden_dim,
                  output_dim=1, num_layers=num_layers) for _ in range(latent_size)])
        self.compress = nn.Linear(lags * (latent_size*2)+1, compress_dim)
        self.compress_dim = compress_dim

    def forward(self, x, z):

        batch_size, length, input_dim = x.shape

        x = x.unfold(dimension=1, size=self.L + 1, step=1)  # [BS, T-L, D, L+1]
        x = torch.swapaxes(x, 2, 3)  # [BS, T-L, L+1, D]
        shape = x.shape
        x = x.reshape(-1, self.L + 1, input_dim)
        xx, yy = x[:, -1:], x[:, :-1]  # yy shape batch_size x 1 x z_dim

        yy = yy.reshape(-1, self.L * input_dim)

        z = z.unfold(dimension=1, size=self.L + 1, step=1)
        z = torch.swapaxes(z, 2, 3)
        z = z.reshape(-1, self.L + 1, input_dim)

        zz1, zz2 = z[:, -1:], z[:, :-1]

        zz1 = zz1.reshape(-1, self.L * (input_dim))

        residuals = []

        hist_jac = []

        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):

            inputs = torch.cat([yy] + [zz1] + [xx[:, :, i]], dim=-1).float()
            if inputs.shape[-1] > 100:
                inputs = self.compress(inputs)
            residual = self.gs[i](inputs)

            with torch.enable_grad():
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant: product of diagonal entries, sum of last entry
            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
            hist_jac.append(torch.unsqueeze(pdd[:, 0, :-1], dim=1))


            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length - self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac

class NPTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, num_layers=3, hidden_dim=64, compress_dim=10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size


        self.gs = nn.ModuleList([MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim,
                                      output_dim=1, num_layers=num_layers) for _ in
                                 range(latent_size)]) if latent_size > 100 else nn.ModuleList(
            [MLP2(input_dim=lags * (latent_size) + 1, hidden_dim=hidden_dim,
                  output_dim=1, num_layers=num_layers) for _ in range(latent_size)])

        self.compress = nn.Linear(lags * latent_size, compress_dim)
        self.compress_dim = compress_dim
        # self.fc = MLP(input_dim=embedding_dim,hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2)

    def forward(self, x, mask=None):
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags

        batch_x = x.unfold(dimension=1, size=self.lags +
                                             1, step=1).transpose(2, 3)
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)
        batch_x_lags = batch_x[:, :-1]  # (batch_size x length, lags, x_dim)
        batch_x_t = batch_x[:, -1]  # (batch_size*length, x_dim)

        batch_x_lags = batch_x_lags.reshape(-1, self.lags * x_dim)
        if x.shape[-1] > 100:
            batch_x_lags = self.compress(batch_x_lags)
        sum_log_abs_det_jacobian = 0
        residuals = []
        for i in range(self.latent_size):
            # (batch_size x length, hidden_dim + lags*x_dim + 1)

            if mask is not None:
                batch_inputs = torch.cat(
                    (batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat(
                    (batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # (batch_size x length, 1)

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()
            logabsdet = torch.log(torch.abs(data_J[:, -1]))

            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)
        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, length, x_dim)

        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian

class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.lags = args.lags
        depth = 10
        encoder = TSEncoderTime(input_dims=args.seq_len,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=depth) 
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.regressor_time = nn.Linear(320+args.seq_len, args.pred_len).to(self.device)
        
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=depth) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        encoder1 = TSEncoder(
            # args,
            input_dims=args.seq_len,
            output_dims=320,  # standard ts2vec backbone value
            hidden_dims=64,  # standard ts2vec backbone value
            depth=args.depth)
        self.encoder1 = TS2VecEncoderWrapper(encoder1, mask='all_true').to(self.device)

        self.obs_transition_prior = NPObsLatentTransitionPrior(lags=self.args.lags,
                                                               latent_size=self.args.enc_in).to(self.device)
        self.stationary_transition_prior = NPTransitionPrior(lags=args.lags,
                                                             latent_size=args.enc_in,
                                                             num_layers=1,
                                                             hidden_dim=args.hidden_dim).to(self.device)
        self.dim = args.c_out * args.pred_len
        self.zs_rec = nn.Linear(320, 2 * args.seq_len).to(self.device)
        self.regressor = nn.Linear(320, self.dim).to(self.device)
        self.register_buffer('stationary_dist_mean', torch.zeros(self.args.enc_in).to(self.device))
        self.register_buffer('stationary_dist_var', torch.eye(self.args.enc_in).to(self.device))
        self.sigmoid = nn.Sigmoid()
        
    def forward_weight(self, x, x_mark, g1, g2 , is_training):

        zs_ = self.encoder1.encoder.forward_time(x)
        zs_ = self.zs_rec(zs_)
        zs_rec_mean, zs_rec_std = zs_[:, :, :self.args.seq_len], self.sigmoid(zs_[:, :, -self.args.seq_len:])
        if is_training:
            zs_rec = self.__reparametrize(zs_rec_mean, zs_rec_std)

        else:
            zs_rec = zs_rec_mean

        if is_training:

            zs_kl_loss = self.kl_loss(zs_rec_mean.permute(0, 2, 1),
                                      zs_rec_std.permute(0, 2, 1),
                                      zs_rec.permute(0, 2, 1))

            obs_kl_loss = self.obs_kl_loss(x, zs_rec_mean.permute(0, 2, 1), zs_rec_std.permute(0, 2, 1),
                                           zs_rec.permute(0, 2, 1))

            other_loss = zs_kl_loss * self.args.zc_kl_weight + obs_kl_loss * self.args.obs_weight
        else:
            other_loss = 0

        rep = self.encoder_time.encoder(x)
        rep = torch.cat([zs_rec,rep],dim=-1)
        y = self.regressor_time(rep).transpose(1, 2)
        y1 = rearrange(y, 'b t d -> b (t d)')

        x = torch.cat([x, x_mark], dim=-1)
        rep2 = self.encoder(x)[:, -1]
        y2 = self.regressor(rep2)

        return y1.detach() * g1 + y2.detach() * g2, y1, y2 ,other_loss

    def store_grad(self):
        for name, layer in self.encoder.named_modules():
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()

    def __reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    @property
    def stationary_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def kl_loss(self, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.stationary_transition_prior(z_est)
        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                              lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()
        loss = (kld_normal + kld_laplace)
        return loss

    def obs_kl_loss(self,x, mus, logvars, z_est):
        lags_and_length = z_est.shape[1]
        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()

        residuals, logabsdet, _ = self.obs_transition_prior(
            x,
            z_est
        )

        log_qz_laplace = log_qz[:, self.lags:]

        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(
            residuals), dim=1) + logabsdet
        kld_laplace = (
                              torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (
                              lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()

        return kld_normal + kld_laplace

class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.individual = args.individual
        self.model = net(args, device = self.device)
        self.buffer = Buffer(10, self.device)       
        self.count = 0
        if self.individual:
            self.decision = MLP(n_inputs=args.pred_len * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh()).to(self.device)
            self.weight = torch.zeros(args.enc_in, device = self.device)
            self.bias = torch.zeros(args.enc_in, device = self.device)
        else:
            self.decision = MLP(n_inputs=(args.c_out * args.pred_len) * 3, n_outputs=1, mlp_width=32, mlp_depth=3, mlp_dropout=0.1, act=nn.Tanh()).to(self.device)
            self.weight = torch.zeros(1, device = self.device)
            self.bias = torch.zeros(1, device = self.device)
        self.weight.requires_grad = True
         
        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        
        # setting = '{}_{}_pl{}_ol{}_opt{}_tb{}'.format(self.args.method, self.args.data, self.args.pred_len,self.args.online_learning, self.args.opt, self.args.test_bsz)
        # folder_path = './results{}/{}/'.format(self.args.n_inner, setting)
        
        # wandb.init(
        #     dir=folder_path,
        #     project='fsnet',
        #     entity=setting,
        #     name=self.args.method,
        # )
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        self.opt_w = optim.Adam([self.weight], lr=self.args.learning_rate_w)
        self.opt_bias = optim.Adam(self.decision.parameters(), lr=self.args.learning_rate_bias)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            loss_ws, loss_biass = [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                self.opt.zero_grad()
                pred, true, loss_w, loss_bias,other_loss = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred[0], true)+ criterion(pred[1], true)
                loss = loss + other_loss
                train_loss.append(loss.item())
                loss_ws.append(loss_w)
                loss_biass.append(loss_bias)
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            loss_ws = np.average(loss_ws)
            loss_biass = np.average(loss_biass)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f} Test Loss:  {4:.4f} loss_ws:  {5:.4f} loss_bias:  {6:.4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, loss_ws, loss_biass))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)
            # adjust_learning_rate(self.opt_w, epoch + 1, self.args)
            # adjust_learning_rate(self.opt_bias, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true, _, _,other_loss = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            pred = pred[0] * 0.5 + 0.5 * pred[1]
            loss = criterion(pred.detach().cpu(), true.detach().cpu()) + other_loss
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        if self.individual:
            self.weight = torch.zeros(self.args.enc_in, device = self.device)
            self.bias = torch.zeros(self.args.enc_in, device = self.device)
        else:
            self.weight = torch.zeros(1, device = self.device)
            self.bias = torch.zeros(1, device = self.device)
        self.weight.requires_grad = True
        self.opt_w = optim.Adam([self.weight], lr=self.args.learning_rate_w)
        

        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()
        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False 
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False

        preds = []
        trues = []
        x_list = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.reshape(self.args.pred_len, self.args.enc_in).detach().cpu())  # (1, pred_len * enc_ibn)
            trues.append(true.reshape(self.args.pred_len, self.args.enc_in).detach().cpu())
            x_list.append(torch.squeeze(batch_x).detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)
        os.makedirs(f"./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/", exist_ok=True)
        # np.save(f'./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/pred.npy', np.array(preds))
        # np.save(f'./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/y.npy', np.array(trues))
        # np.save(f'./draw_data/{self.args.method}/{self.args.data}/{self.args.pred_len}/x.npy', np.array(x_list))
        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        
        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        # print(self.weight[0], self.bias[0])
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device) #torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()
        
        b, t, d = batch_y.shape
        
        if self.individual:
            loss1 = F.sigmoid(self.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
            loss1 = rearrange(loss1, 'b t d -> b (t d)')
        else:
            loss1 = F.sigmoid(self.weight)
        if mode == 'train':
            is_training = True
        else:
            is_training = False
        outputs, y1, y2, other_loss = self.model.forward_weight(x, batch_x_mark, loss1, 1 - loss1, is_training)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        
        b, t, d = batch_y.shape
        criterion = self._select_criterion()
        
        l1, l2 = criterion(y1, rearrange(batch_y, 'b t d -> b (t d)')), criterion(y2, rearrange(batch_y, 'b t d -> b (t d)'))
        
        loss_w = criterion(outputs, rearrange(batch_y, 'b t d -> b (t d)'))
        loss_w.backward()
        self.opt_w.step()   
        self.opt_w.zero_grad()   
        
        if self.individual:
            y1_w, y2_w = y1.view(b, t, d).detach(), y2.view(b, t, d).detach()
            true_w = batch_y.view(b, t, d).detach()
            loss1 = F.sigmoid(self.weight).view(1, 1, -1)
            loss1 = loss1.repeat(b, t, 1)
            
            inputs_decision = torch.cat([loss1*y1_w, (1-loss1)*y2_w, true_w], dim=1)
            
            self.bias = self.decision(inputs_decision.permute(0,2,1))
            weight = self.weight.view(1, 1, -1)
            weight = weight.repeat(b, t, 1)
            bias = self.bias.view(b, 1, -1)
            loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
            loss1 = rearrange(loss1, 'b t d -> b (t d)')
            loss2 = 1 - loss1
            
            y1_w = rearrange(y1_w, 'b t d -> b (t d)')
            y2_w = rearrange(y2_w, 'b t d -> b (t d)')
            true_w = rearrange(true_w, 'b t d -> b (t d)')
        else:
            y1_w, y2_w = y1.view(b, t * d).detach(), y2.view(b, t * d).detach()
            true_w = batch_y.view(b, t * d).detach()
            loss1 = F.sigmoid(self.weight)  
            inputs_decision = torch.cat([loss1*y1_w, (1-loss1)*y2_w, true_w], dim=1)
            self.bias = self.decision(inputs_decision)
            loss1 = F.sigmoid(self.weight + self.bias)
            loss2 = 1 - loss1
        
        loss_bias = criterion(loss1 * y1_w + loss2 * y2_w, true_w)
        loss_bias.backward()
        self.opt_bias.step()   
        self.opt_bias.zero_grad()   
        
        return [y1, y2], rearrange(batch_y, 'b t d -> b (t d)'), loss_w.detach().cpu().item(), loss_bias.detach().cpu().item(), other_loss
    
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, return_loss=False):
        b, t, d = batch_y.shape
        is_training = True
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        for _ in range(self.n_inner):
            
            if self.individual:
                weight = self.weight.view(1, 1, -1)
                weight = weight.repeat(b, t, 1)
                bias = self.bias.view(-1, 1, d)
                loss1 = F.sigmoid(weight + bias.repeat(1, t, 1)).view(b, t, d)
                loss1 = rearrange(loss1, 'b t d -> b (t d)')
            else:
                loss1 = F.sigmoid(self.weight + self.bias)
                
            outputs, y1, y2, other_loss = self.model.forward_weight(x, batch_x_mark, loss1, 1-loss1, is_training)

            l1, l2 = criterion(y1, true), criterion(y2, true)
            loss = l1 + l2 + other_loss
            loss.backward()
            self.opt.step()    
            self.opt.zero_grad()
            
            if self.individual:
                y1_w, y2_w = y1.view(b, t, d).detach(), y2.view(b, t, d).detach()
                true_w = batch_y.view(b, t, d).detach()
                loss1 = F.sigmoid(self.weight).view(1, 1, -1)
                loss1 = loss1.repeat(b, t, 1)
                inputs_decision = torch.cat([loss1*y1_w, (1-loss1)*y2_w, true_w], dim=1)
                self.bias = self.decision(inputs_decision.permute(0,2,1))
                weight = self.weight.view(1, 1, -1)
                weight = weight.repeat(b, t, 1)
                bias = self.bias.view(b, 1, -1)
                loss1 = F.sigmoid(weight + bias.repeat(1, t, 1))
                loss1 = rearrange(loss1, 'b t d -> b (t d)')
                loss2 = 1 - loss1
                
                y1_w = rearrange(y1_w, 'b t d -> b (t d)')
                y2_w = rearrange(y2_w, 'b t d -> b (t d)')
                true_w = rearrange(true_w, 'b t d -> b (t d)')
            else:
                y1_w, y2_w = y1.view(b, t * d).detach(), y2.view(b, t * d).detach()
                true_w = batch_y.view(b, t * d).detach()
                loss1 = F.sigmoid(self.weight)
                inputs_decision = torch.cat([loss1*y1_w, (1-loss1)*y2_w, true_w], dim=1)
                self.bias = self.decision(inputs_decision)
                loss1 = F.sigmoid(self.weight + self.bias)
                loss2 = 1 - loss1
            
            outputs_bias = loss1 * y1_w + loss2 * y2_w
            loss_bias = criterion(outputs_bias, true_w)
            loss_bias.backward()
            self.opt_bias.step()   
            self.opt_bias.zero_grad()   
            
            if self.individual:
                loss1 = F.sigmoid(self.weight).view(1, 1, -1)
                loss1 = loss1.repeat(b, t, 1)
                loss1 = rearrange(loss1, 'b t d -> b (t d)')
            else:
                loss1 = F.sigmoid(self.weight)  
            loss_w = criterion(loss1 * y1.detach() + (1 - loss1) * y2.detach(), rearrange(batch_y, 'b t d -> b (t d)'))
            loss_w.backward()
            self.opt_w.step()   
            self.opt_w.zero_grad()   
            
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        idx = self.count +  torch.arange(batch_y.size(0)).to(self.device)
        self.count += batch_y.size(0)
        self.buffer.add_data(examples = x, labels = true, logits = idx, task_labels=batch_x_mark)
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')

