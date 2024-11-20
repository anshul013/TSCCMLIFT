import argparse
import os
import random
import numpy as np
import torch
import math
from exp.exp_main import Exp_Main
from exp.exp_ccm import Exp_CCM
# import wandb


fix_seed = 3000
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.autograd.set_detect_anomaly(True)
np.random.seed(fix_seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TSMixer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# DLinear
# parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of MLP in transformer')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# CCM specific arguments
parser.add_argument('--zero_shot_test', type=bool, default=False)
parser.add_argument('--test_data', type=str, default='ETTm2', help='data for testing')
parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=48, help='output MTS length (τ)')
parser.add_argument('--individual', type=str, default="c", help="i: individual; c: cluster, else: all-in dimension")
parser.add_argument('--beta', type=float, default=0.3, help="loss weight for similarity loss")
parser.add_argument('--n_layers', type=int, default=4, help='num of encoder layers (N)')
parser.add_argument('--cluster_ratio', type=float, default=0.3, help="ratio of clusters")
parser.add_argument('--attn_dropout', type=float, default=0.3, help='attention dropout')
parser.add_argument('--pre_norm', type=bool, default=False, help='pre normalization')
parser.add_argument('--pretrain_head', type=bool, default=False, help='pretrain head')
parser.add_argument('--patch_len', type=int, default=16, help='patch length (L_seg)')
parser.add_argument('--max_seq_len', type=int, default=1024, help="maximum number of sequence_length")
parser.add_argument('--padding_patch', type=str, default='end', help='None: None; end: padding on the end')

# Mixers
parser.add_argument('--num_blocks', type=int, default=3, help='number of mixer blocks to be used in TSMixer')
parser.add_argument('--hidden_size', type=int, default=32, help='first dense layer diminsions for mlp features block')
parser.add_argument('--single_layer_mixer', type=str2bool, nargs='?', default=False, help="if true a single layer mixers are used")

#Common between Mixers and Transformer-based models
parser.add_argument('--activation', type=str, choices={"gelu", "relu", "linear"}, default='gelu', help='activation')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--early_stopping', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="whether to include early stopping or not")
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--norm', type=str, choices={"batch", "instance"}, default="batch", help="type of normalization")

# Patching and Convolution models
parser.add_argument('--exclude_inter_patch_mixing', type=str2bool, default=False, help='define if inter patch mixing is used for PatchTSMixer model')
parser.add_argument('--exclude_intra_patch_mixing', type=str2bool, default=False, help='define if intra patch mixing is used for PatchTSMixer model')
parser.add_argument('--exclude_channel_mixing', type=str2bool, default=False, help='define if channel mixing is used for PatchTSMixer model')
parser.add_argument('--patch_size', type=int, default=16, help="Number of timesteps per patch")
parser.add_argument('--kernel_size', type=int, default=1, help="conv width to cover certain number of timesteps")
parser.add_argument('--stride', type=int, default=1, help='number of non-overlapping timesteps for conv operation')
parser.add_argument('--affine', type=str2bool, default=True, help='define if the rev_norm is affine or not')
parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension for models including patch embedding")

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument("--use_gpu", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="use gpu")
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# SciNet
parser.add_argument("--num_levels", type=int, default=3)
parser.add_argument("--num_decoder_layer", type=int, default=1)
parser.add_argument("--concat_len", type=int, default=0)
parser.add_argument("--groups", type=int, default=3)
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--single_step_output_One", type=int, default=3)
parser.add_argument("--positionalE",  type=str2bool, nargs='?', const=False, default=False)
parser.add_argument("--modified",  type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--RIN",  type=str2bool, nargs='?', const=False, default=False)


args = parser.parse_args()

# Data parser dictionary
data_parser = {
    'ETTh1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    'ETTm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'ETTh2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[12*30*24, 4*30*24, 4*30*24]},
    'ETTm2':{'data':'ETTm2.csv', 'data_dim':7, 'split':[4*12*30*24, 4*4*30*24, 4*4*30*24]},
    'WTH':{'data':'WTH.csv', 'data_dim':12, 'split':[0.7, 0.1, 0.2]},
    'ECL':{'data':'ECL.csv', 'data_dim':321, 'split':[0.7, 0.1, 0.2]},
    'ILI':{'data':'ILI.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
    'TRF':{'data':'TRF.csv', 'data_dim':862, 'split':[0.7, 0.1, 0.2]},
    'EXR':{'data':'EXR.csv', 'data_dim':8, 'split':[0.7, 0.1, 0.2]},
}

# Process data information
if args.zero_shot_test is False:
    args.test_data = args.data

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']

args.freq = "t" if args.data in ["ETTm1", "ETTm2"] else "h"
if args.test_data in data_parser.keys():
    args.test_data_path = data_parser[args.test_data]['data']
    args.test_data_split = data_parser[args.test_data]['split']

args.n_cluster = math.ceil(args.data_dim * args.cluster_ratio)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

# Select experiment class based on model
if args.model == 'TSMixerC':
    Exp = Exp_CCM
else:
    Exp = Exp_Main

# wandb.login(key="")  # Add your wandb key here if needed
# wandb.init(project="time_series_forecasting")  # You can change the project name
# wandb.config.update(args)

if args.is_training:
    for ii in range(args.itr):
       # setting record of experiments
        if args.model == 'TSMixerC':
            setting = '{}_{}_il{}_ol{}_pl{}_ratio{}_dm{}_nh{}_el{}_itr{}'.format(
                args.model_id,
                args.model,
                args.in_len,
                args.out_len,
                args.patch_len,
                args.cluster_ratio,
                args.d_model,
                args.n_heads,
                args.n_layers,
                ii)
        else:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
