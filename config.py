import argparse

#conda activate sss
#cd /media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/hxm/mutil_view
#nohup python train_slice.py

parser = argparse.ArgumentParser(description='Hyper-parameters management')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')

parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--save', default='output_3/brats18/model07/', help='save path of trained model')
# parser.add_argument('--save',default='/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/ljy/mean_teacher/output/HPF/model9/',help='save path of trained model')

parser.add_argument('--resume',
                    default='/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/hxm/Duo_MT/output/result/12/epoch100.pth',
                    help='save path of trained model')

parser.add_argument('--save_nii', default='output_3/brats18/model07/', help='save path of result')

parser.add_argument('--name', default='brats')

parser.add_argument('--test_txt', default='data_brats_TCI_60_c_test.txt', help='test txt name')
parser.add_argument('--test_path', default='/media/root/18TB_HDD/hxm/hxm/mutil_view/data/train_dataset/', help='test path name')

parser.add_argument('--dataset_path_1',default='/media/root/18TB_HDD/hxm/hxm/mutil_view/data/train_dataset/')
parser.add_argument('--dataset_path_2',default='/media/root/18TB_HDD/hxm/hxm/mutil_view/data/train_dataset_a/')

parser.add_argument('--label_txt_1', default='data_brats_TCI_80_c_train.txt')
parser.add_argument('--label_txt_2', default='data_brats_TCI_80_a_train.txt')


parser.add_argument('--img_size', type=int, default=224, help='patch size of train samples after resize')

parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--channel', type=int, default=1, metavar='N', help='number of channels (default: 1)')

parser.add_argument('--lr_gen', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--early-stop', default=None, type=int, help='early stopping (default: 20)')

parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')

parser.add_argument('--ema_decay', type=float, default=0.5, help='ema_decay')

parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')

parser.add_argument('--consistency', type=float, default=0.1, help='consistency')

parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

parser.add_argument('--batch_size', type=list, default=32, help='batch size of trainset')

parser.add_argument('--labeled_bs', type=int, default=32, help='labeled_batch_size per gpu')
parser.add_argument('--device', type=str, default='cuda')  # 运行设备

parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


