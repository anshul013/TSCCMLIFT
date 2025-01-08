from data_provider.data_loader import Dataset_MTS
from exp.exp_basic import Exp_Basic
from models.hcm.tsmixer import HardClusterTSMixer  # Updated import
import torch.nn.functional as F
from utils.ccm.tools import EarlyStopping, adjust_learning_rate
from utils.ccm.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
from torchinfo import summary

class Exp_HCM(Exp_Basic):
    def __init__(self, args):
        super(Exp_HCM, self).__init__(args)
    
    def _build_model(self):
        model = HardClusterTSMixer(self.args).float()
        summary(model)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size
            data_set = Dataset_MTS(
                root_path=args.root_path,
                data_path=args.test_data_path,
                flag=flag,
                size=[args.in_len, args.out_len],  
                data_split = args.test_data_split,
            )
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
            data_set = Dataset_MTS(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.in_len, args.out_len],  
                data_split = args.data_split,
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def get_similarity_matrix(self, batch_x):
        batch_x = batch_x.to(self.device).float()
        batch_x = batch_x.transpose(1, 2)
        vars_mean = batch_x.mean(0)
        
        diff = vars_mean.unsqueeze(0) - vars_mean.unsqueeze(1)
        dist_squared = torch.sum(diff ** 2, dim=-1)
        
        param = torch.max(dist_squared)
        euc_similarity = torch.exp(-5 * dist_squared / param)
        return euc_similarity.to(self.device)

    def similarity_loss_batch(self, assignments, simMatrix):
        # assignments: [n_vars, n_cluster] one-hot matrix
        # simMatrix: [n_vars, n_vars]
        temp_1 = torch.mm(assignments.t(), simMatrix)
        SAS = torch.mm(temp_1, assignments)
        _SS = 1 - torch.mm(assignments, assignments.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix))
        return loss

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        loss_f_list = []
        loss_s_list = []
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true, assignments = self._process_one_batch(vali_data, batch_x, batch_y)
                loss_f = nn.MSELoss()(pred.detach().cpu(), true.detach().cpu())
                
                simMatrix = self.get_similarity_matrix(batch_x)
                loss_s = self.similarity_loss_batch(assignments, simMatrix)
                
                loss = loss_f + self.args.beta * loss_s
                total_loss.append(loss.item())
                loss_f_list.append(loss_f.item())
                loss_s_list.append(loss_s.item())
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)

        total_loss = np.average(total_loss)
        loss_f = np.average(loss_f_list)
        loss_s = np.average(loss_s_list)
        
        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num
        
        return metrics_mean[1], total_loss, loss_f, loss_s, metrics_mean[0]  # mse, total_loss, forecast_loss, similarity_loss, mae

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            tl_f = []
            tl_s = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                pred, true, assignments = self._process_one_batch(train_data, batch_x, batch_y)
                loss_f = criterion(pred, true)
                
                simMatrix = self.get_similarity_matrix(batch_x)
                loss_s = self.similarity_loss_batch(assignments, simMatrix)
                
                loss = loss_f + self.args.beta * loss_s
                train_loss.append(loss.item())
                tl_f.append(loss_f.item())
                tl_s.append(loss_s.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            
            vali_mse, vali_loss, vali_loss_f, vali_loss_s, vali_mae = self.vali(vali_data, vali_loader)
            test_mse, test_loss, test_loss_f, test_loss_s, test_mae = self.vali(test_data, test_loader)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, load=False):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true, _ = self._process_one_batch(test_data, batch_x, batch_y)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num
        
        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        
        return mae, mse, rmse, mape, mspe

    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs, assignments = self.model(batch_x, return_clusters=True)
        return outputs, batch_y, assignments