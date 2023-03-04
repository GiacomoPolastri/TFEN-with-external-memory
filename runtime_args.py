"""
    
    Configurations
    
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_size', type=int, default=224, 
                    help='size of the input file (default 224)')
parser.add_argument('--workers', default=16, type=int,
                    help='Number of data loading workers (default 16)')
parser.add_argument('--epochs', default = 301, type=int, 
                    help = 'Number of epochs to run (default 301')
parser.add_argument('--batch_size', default = 128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default = 0.01, type=float, 
                    help = 'learning rate (default 0.01)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum  (default 0.9)')
parser.add_argument('-wd', '--weight-decay', default=5e-5, type=float,
                    help='weight decay (default: 5e-5)',
                    dest='weight_decay')
parser.add_argument('-da', '--data_augmentation', default = 0.9, type = float,
                    help = 'value of data augmentation (default: 0.9)')
parser.add_argument('-d','--dataset', default = 'Imagenet', type = str, metavar= 'D',
                    help = 'dataset (default: Imagenet)')
parser.add_argument('--mem','--memorize_acc_loss', default = 50, type = int, metavar='MEM',
                    help = 'Number of epoch to wait for memorize acuracy and loss (default:50)')
parser.add_argument('-r', '--resume', default = False, type=bool, metavar='R',
                    help='Resume training')
parser.add_argument('-se','--startepoch', default =1, type=int, 
                    help='Starting epoch (default 1')
parser.add_argument('-cpu', '--cpu', default=False, type=bool,
                    help='boolean for only cpu usage (default False)') 
parser.add_argument('-n', '--number_train', default=0, type=int, 
                    help='number of training (default 0)')
parser.add_argument('-layers', '--layers', default=False, type=bool,
                    help='get all convolutional layers (default False)')
parser.add_argument('-mnet', '--mnet', default=False, type=bool,
                    help=' get mobileNet output (default False)')
parser.add_argument('-cbam', '--cbam', default=False, type=bool,
                    help='get cbam output (default False')
parser.add_argument('-sam', '--sam', default=False, type=bool,
                    help='get sam output (default False')
parser.add_argument('-cam', '--cam', default=False, type=bool,
                    help='get cam output (default False')
parser.add_argument('-last', '--last_cnn_layer', default=False, type=bool,
                    help=' gets the last cnn layer (default False)')
parser.add_argument('-n_det', '--number_detections', default=10, type=int, 
                    help = "number of object inside an image (default 10)")

args = parser.parse_args()