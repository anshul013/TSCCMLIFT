from data_provider.data_loader import Dataset_MTS
from exp.exp_basic import Exp_Basic
from models.hcm1.tsmixer import TSMixerH
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.ccm.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import os
import time
import json
import pickle
from torchinfo import summary
from torch.cuda.amp import GradScaler

class Exp_HCM(Exp_Basic):
    def __init__(self, args):
        super(Exp_HCM, self).__init__(args)
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        model_dict = {
            'TSMixerH': TSMixerH
        }
        model = model_dict[self.args.model](self.args).float()
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
                data_split=args.test_data_split,
            )
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
            data_set = Dataset_MTS(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.in_len, args.out_len],  
                data_split=args.data_split,
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

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, if_update=False)
                # pred shape: [32, 96, 7], true shape: [32, 96, 7]
                loss = nn.MSELoss()(pred.detach().cpu(), true.detach().cpu())
                
                total_loss.append(loss.item())
                
                batch_size = pred.shape[0]
                instance_num += batch_size
                
                # Convert to numpy and ensure shapes are correct
                pred_np = pred.detach().cpu().numpy()  # [32, 96, 7]
                true_np = true.detach().cpu().numpy()  # [32, 96, 7]
                
                # Calculate metrics for each sample in batch
                batch_metrics = []
                for j in range(batch_size):
                    sample_metrics = metric(pred_np[j], true_np[j])  # Calculate for each sample
                    batch_metrics.append(sample_metrics)
                
                # Convert to numpy array and sum
                batch_metrics = np.array(batch_metrics).sum(axis=0)
                metrics_all.append(batch_metrics)

            total_loss = np.average(total_loss)
            metrics_all = np.stack(metrics_all, axis=0)
            metrics_mean = metrics_all.sum(axis=0) / instance_num
            mae, mse, rmse, mape, mspe = metrics_mean
            
        self.model.train()
        return mse, total_loss, mae
                    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # Update clusters periodically
                if_update = (epoch % self.args.cluster_update_interval == 0)
                
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, if_update=if_update)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            vali_mse, vali_loss, vali_mae = self.vali(vali_data, vali_loader)
            test_mse, test_loss, test_mae = self.vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Vali MSE: {:.7f}, MAE: {:.7f}, Test MSE: {:.7f}, MAE: {:.7f}".format(
                vali_mse, vali_mae, test_mse, test_mae))
            
            if epoch % self.args.cluster_update_interval == 0:
                print("Current cluster assignments:", self.model.get_current_assignments())
                metrics = self.model.get_clustering_metrics()
                if metrics['inertia'] is not None:
                    print("Clustering inertia:", metrics['inertia'])
                if metrics['silhouette'] is not None:
                    print("Silhouette score:", metrics['silhouette'])

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        # Load best model using our custom method with initialization
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Get first batch for initialization
        # first_batch = next(iter(train_loader))[0]
        # state_dict = torch.load(best_model_path)
        # self.model.load_state_dict_with_init(state_dict, first_batch)
        
        # Handle DataParallel and save state
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path + '/' + 'checkpoint.pth')
        
        return self.model

    def test(self, setting, save_pred=True, inverse=False):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse, if_update=False)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # Save results
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False, if_update=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        
        outputs = self.model(batch_x, if_update=if_update)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)
        return outputs, batch_y

    def eval(self, setting, save_pred=True, inverse=False):
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],  
            data_split=args.data_split,
            scale=True,
            scale_statistic=args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse, if_update=False)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # Save results
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('Evaluation metrics - MSE: {}, MAE: {}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)
        return 