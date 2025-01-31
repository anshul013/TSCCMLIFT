from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.hcm1.tsmixer import TSMixerH, TMixerH
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.ccm.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import json
from torchinfo import summary

class Exp_HCM(Exp_Basic):
    def __init__(self, args):
        super(Exp_HCM, self).__init__(args)
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        model_dict = {
            'TSMixerH': TSMixerH,
            'TMixerH': TMixerH
        }
        model = model_dict[self.args.model](self.args).float()
        
        # Get initial data for clustering
        train_data, train_loader = self._get_data(flag='train')
        full_data = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                full_data.append(batch_x)
        full_data = torch.cat(full_data, dim=0).to(self.device)
    
        # Initialize clusters first
        print("Initializing clusters with full training data...")
        model.initialize_clusters(full_data)
        
        # Print model summary after cluster initialization
        print("\nModel Architecture:")
        summary(model)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        print(flag, len(data_set))
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
                loss = nn.MSELoss()(pred, true)
                total_loss.append(loss.item())
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                
                # Calculate metrics on GPU
                metrics = torch.stack([
                    torch.mean(torch.abs(pred - true)),  # MAE
                    torch.mean((pred - true) ** 2),  # MSE
                    torch.sqrt(torch.mean((pred - true) ** 2)),  # RMSE
                    torch.mean(torch.abs((pred - true) / true)),  # MAPE
                    torch.mean((pred - true) ** 2 / true ** 2)  # MSPE
                ]) * batch_size
                metrics_all.append(metrics)

            total_loss = np.average(total_loss)
            metrics_all = torch.stack(metrics_all, dim=0)
            metrics_mean = metrics_all.sum(dim=0) / instance_num
            mae, mse, rmse, mape, mspe = metrics_mean.tolist()
            
        self.model.train()
        return mse, total_loss, mae

    def train(self, setting):
        print(f"\nTraining on device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = nn.MSELoss()

        # Get full training data for clustering
        # print("Collecting full training data for clustering initialization...")
        # full_data = []
        # with torch.no_grad():
        #     for i, (batch_x, batch_y) in enumerate(train_loader):
        #         full_data.append(batch_x)
        # full_data = torch.cat(full_data, dim=0).to(self.device)
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        #     full_data.append(batch_x)
        # full_data = torch.cat(full_data, dim=0)  # [total_samples, seq_len, channels]
    
        # Initialize clusters with full data once
        # print("Initializing clusters with full training data...")
        # self.model.initialize_clusters(full_data)
        
        # Save model architecture and initial settings
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        
        # Print initial cluster assignments
        print("Initial cluster assignments:", self.model.get_current_assignments())
        metrics = self.model.get_clustering_metrics()
        print("Initial clustering metrics:", metrics)
        
        print("\nStart training: {} epochs".format(self.args.train_epochs))
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            self.model.train()
            train_loss = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                pred, true = self._process_one_batch(train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()

                if i % 100 == 0:
                    print(f"\titr: {i:03d}, loss: {loss.item():.7f}")
                    if torch.cuda.is_available():
                        print(f"\tGPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_mse, vali_loss, vali_mae = self.vali(vali_data, vali_loader)
                test_mse, test_loss, test_mae = self.vali(test_data, test_loader)
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                print("Vali MSE: {:.7f}, MAE: {:.7f}, Test MSE: {:.7f}, MAE: {:.7f}".format(
                    vali_mse, vali_mae, test_mse, test_mae))

                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
    
        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        # Get first batch for initialization
        # first_batch = next(iter(train_loader))[0]
        state_dict = torch.load(best_model_path)
        self.model.load_state_dict(state_dict)
        # self.model.load_state_dict_with_init(state_dict, first_batch)
        
        # Handle DataParallel and save state
        # state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        # torch.save(state_dict, path + '/' + 'checkpoint.pth')
        
        return self.model

    def test(self, setting, save_pred=True, inverse=False):
        print("\nEvaluation on test set...")
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                pred, true = self._process_one_batch(test_data, batch_x, batch_y, inverse)
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                
                # Calculate metrics on GPU
                metrics = torch.stack([
                    torch.mean(torch.abs(pred - true)),
                    torch.mean((pred - true) ** 2),
                    torch.sqrt(torch.mean((pred - true) ** 2)),
                    torch.mean(torch.abs((pred - true) / true)),
                    torch.mean((pred - true) ** 2 / true ** 2)
                ]) * batch_size
                metrics_all.append(metrics)
                
                if save_pred:
                    preds.append(pred)
                    trues.append(true)

        metrics_all = torch.stack(metrics_all, dim=0)
        metrics_mean = metrics_all.sum(dim=0) / instance_num
        mae, mse, rmse, mape, mspe = metrics_mean.tolist()

        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('Evaluation metrics - MSE: {}, MAE: {}'.format(mse, mae))

        if save_pred:
            preds = torch.cat(preds, dim=0)
            trues = torch.cat(trues, dim=0)
            np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path+'pred.npy', preds.cpu().numpy())
            np.save(folder_path+'true.npy', trues.cpu().numpy())
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        pred_len = self.args.pred_len
        
        # Forward pass through model
        outputs = self.model(batch_x)
        batch_y = batch_y[:, -pred_len:, :]
        
        if outputs.shape[1] != pred_len:
            outputs = outputs[:, :pred_len, :]
        
        if inverse:
            outputs = dataset_object.inverse_transform(outputs.cpu()).to(self.device)
            batch_y = dataset_object.inverse_transform(batch_y.cpu()).to(self.device)
        
        return outputs, batch_y 