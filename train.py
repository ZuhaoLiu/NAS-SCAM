import os
import sys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from optparse import OptionParser
import torch.nn as nn

from tqdm import tqdm
from network_search import SearchNet
from network_rebuild import RebuildNet
from train_functions import train_net, test_net



def get_args():
    parser = OptionParser()
    parser.add_option('-s', '--search_epochs', dest='search_epochs', default=120, type='int',
            help='number of search epochs')
    parser.add_option('-r', '--rebuild_epochs', dest='rebuild_epochs', default=300, type='int',
            help='number of rebuild epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
            type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-3,
            type='long', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
            default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
            default=False, help='load file model')
    parser.add_option('-d', '--dataset_path', dest='dataset_path',
            default='./final_dataset', help='the dictionary path of the dataset')

    (options, args) = parser.parse_args()
    return options

def save_index(weights_SAM, weights_CAM):
    '''
    we save the searched results as index into a npy file
    For example, if the total number of nodes is 4, i.e. node1 -> node2-> node3 -> node4
    then the shape of weights_SAM or weights_CAM should be (8, 6)
    8 means the network totally has 8 convolution bolcks
    and 6 correspnds to selected operation between each two nodes, as following
    [ops(node1->node2),ops(node1->node3),ops(node1->node4),ops(node2->node3),ops(node2->node4),ops(node3->node4)]
    '''
    weights_SAM = torch.argmax(weights_SAM, dim = -1)
    weights_SAM = weights_SAM.cpu().numpy()
    np.save('SAM.npy', weights_SAM)
    weights_CAM = torch.argmax(weights_CAM, dim = -1)
    weights_CAM = weights_CAM.cpu().numpy()
    np.save('CAM.npy', weights_CAM)

def train_model(args, save_file_name, rebuild = False, SAM_Index = None, CAM_Index = None):
    '''
    base training function
    Args:
        save_file_name: file name to save the test results
        rebuild: if doing rebuild network training, set this flag as True
        SAM_Index: if doing rebuild network training, provides this as the architecture of NAS-SAM
        CAM_Index: if doing rebuild network training, provides this as the architecture of NAS-CAM
    '''
    torch.cuda.empty_cache()
    if rebuild:
        net  = RebuildNet(SAM_Index, CAM_Index, NChannels = 3, NClasses = 4)
        epochs = args.rebuild_epochs
        pth_name = 'RebuildSave.pth'
    else:
        net  = SearchNet(NChannels = 3, NClasses = 4)
        epochs = args.search_epochs
        pth_name = 'SearchSave.pth'
    if args.gpu:
        cudnn.benchmark = True
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    try:
        train_net(net,
            args.dataset_path,
            epochs=epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            gpu=args.gpu,
            rebuild = rebuild
            )
        return test_net(net,
            args.dataset_path,
            save_file_name,
            batch_size=args.batchsize,
            gpu=args.gpu,
            rebuild = rebuild,
            pth_name = pth_name
            )
			
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def search_model(args):
    ''' 
    search the architecture of NAS-SCAM
    '''
    weights_SAM, weights_CAM = train_model(args, save_file_name = 'search_results.txt')
    save_index(weights_SAM, weights_CAM)

def rebuild_model(args, number):
    '''
    rebuild the network based on searched results
    '''
    SAM_Index = np.load('SAM.npy')
    CAM_Index = np.load('CAM.npy')
    total_dice = np.zeros([number, 4])
    avg_dice = np.zeros([number])
    average_ACC = np.zeros([number])
    for i in range(number): # do the experiments repeatedly, and get the average results
        total_dice[i], avg_dice[i], average_ACC[i] = train_model(args, save_file_name = 'rebuild_results.txt', 
                                    rebuild = True, SAM_Index = SAM_Index, CAM_Index = CAM_Index)
    # write the results into the txt file
    print('\naverage dice: {:.4f} +- {:.4f}, average accuracy: {:.4f} +- {:.4f}'.format(np.mean(avg_dice), np.std(avg_dice), np.mean(average_ACC), np.std(average_ACC)))
    for i in range(4):
        print("class{:.0f}'s dice: {:.4f} +- {:.4f}".format(i+1, np.mean(total_dice[:,i]), np.std(total_dice[:,i])))

    f = open('results/average_rebuild_result.txt', 'a')	
    f.write('average dice: {:.4f} +- {:.4f}, average accuracy: {:.4f} +- {:.4f}\n'.format(np.mean(avg_dice), np.std(avg_dice), np.mean(average_ACC), np.std(average_ACC)))
    for i in range(4):
        f.write("class{:.0f}'s dice: {:.4f} +- {:.4f}\n".format(i+1, np.mean(total_dice[:,i]), np.std(total_dice[:,i])))
    f.write('\n\n')
    f.close()


if __name__ == '__main__':
    args = get_args()
    search_model(args)
    rebuild_model(args, 1)





