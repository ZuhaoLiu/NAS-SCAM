import os
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Function, Variable
import numpy as np
from optparse import OptionParser
import torch.nn as nn
from PIL import Image
import math
import nibabel as nib
from tqdm import tqdm
import time
import random
from crop_recover import crop_image, recover_image
from utils import data_aug



def get_total_batch(batch_size, total_length):
    '''
    calculate batch number for a epoch
    '''
    return int(math.ceil(total_length/float(batch_size)))

def get_batch(image, batch_size):
    '''
    output a batch of data
    '''
    index = 0
    def output_batch():
        nonlocal index
        output = image[index*batch_size:(index+1)*batch_size]
        index += 1
        return output
    return output_batch

def shuffle_set(image, label):
    per = np.random.permutation(image.shape[0])
    new_image = image[per,:,:,:]
    new_label = label[per,:,:,:]
    return new_image, new_label


class dice_loss(nn.Module):
    '''
    normal dice loss
    '''
    def __init__(self):
        super(dice_loss, self).__init__()
    def forward(self, logits, label, smooth = 1.):
        intersection = torch.sum(logits * label, dim = (2,3), keepdim = True)
        p = torch.sum(logits, dim = (2,3), keepdim = True)
        t = torch.sum(label, dim = (2,3), keepdim = True)
        in_loss = torch.mean((2 * intersection)/(p + t + smooth))
        dice_loss = 1 - in_loss
        return dice_loss

class weighted_log_cross(nn.Module):
    '''
    weighted logarithm cross entropy loss
    '''
    def __init__(self, w_dice = 0.5, w_cross = 0.5, image_width = 256, image_height = 256):
        super(weighted_log_cross, self).__init__()
        self.w_dice = w_dice
        self.w_cross = w_cross
        self.image_pixel_number = image_width * image_height
    def forward(self, logits, label,smooth = 1.):
        weights_number = torch.sum(label, dim = (2,3), keepdim = True)/self.image_pixel_number
        weights = label*(1-weights_number)
        weights = weights + (1-label)*weights_number
        cross = torch.mean(-weights*label*torch.log(logits+1e-9),dim = (0,2,3), keepdim = True)
        cross = torch.mean(cross)
        area_union = torch.sum(logits * label, dim = (0,2,3), keepdim = True)
        area_logits = torch.sum(logits, dim = (0,2,3), keepdim = True)
        area_label = torch.sum(label, dim = (0,2,3), keepdim = True)
        dice = torch.mean(torch.pow((-1) * torch.log((2 * area_union + 1e-7)/(area_logits + area_label + smooth)), 0.3))
        loss = self.w_dice*dice + self.w_cross*cross
        return loss


class log_loss(nn.Module):
    '''
    logarithm exponential dice loss
    '''
    def __init__(self, w_dice = 0.5, w_cross = 0.5):
        super(log_loss, self).__init__()
        self.w_dice = w_dice
        self.w_cross = w_cross
    def forward(self, logits, label, smooth = 1.):
        area_union = torch.sum(logits * label, dim = (0,2,3), keepdim = True)
        area_logits = torch.sum(logits, dim = (0,2,3), keepdim = True)
        area_label = torch.sum(label, dim = (0,2,3), keepdim = True)
        in_dice = torch.mean(torch.pow((-1) * torch.log((2 * area_union + 1e-7)/(area_logits + area_label + smooth)), 0.3))
        return in_dice

def get_seperate_areas(inputs, target):
    '''
    Intermediate function to calculate the dice accurately
    calculate the areas instead of dice directly can avoid the error caused by no-foreground prediction
    '''
    areas = np.zeros([4,2]).astype(np.float32)
    for number in range(4):
        inter = np.dot(inputs[:,:,number].flatten(), target[:,:,number].flatten())
        union = np.sum(inputs[:,:,number]) + np.sum(target[:,:,number])
        areas[number, 0] = inter
        areas[number, 1] = union
    return areas

def calculate_dice(areas):
    '''
    calculate dice coefficient based on areas
    '''
    dice = np.zeros([4]).astype(np.float32)
    for number in range(4):
        in_dice = 2 * areas[number, 0] / areas[number, 1]
        dice[number] = in_dice
    return dice

def yield_data(dataset_path, aug_name = ['gaussian_blur'], aug_proportion = [0.2], search = False):
    '''
    we empirically found that add same number of images for each class in one batch can significantly increase the accuracy
    the batch size in the searching process is 4, and 1 for each class (totally 4 classes)
    Args:
        dataset_path: dictionary path of the dataset
        aug_name: add augmentation name in the list, the supported methods including 'random_rotate', 'random_flip'
            'gaussian_blur', 'median_blur' and 'elastic_transform'
        aug_proportion: the probability to do each of the augmentation method, the length of this list must be the 
            same as the aug_name
        search: if this flag is True, the data is generated for searching, if False, the data is generated for rebuilding
    '''
    batch_image_numpy = np.zeros([4, 256, 256, 3])
    batch_label_numpy = np.zeros([4, 256, 256, 4])
    if search:
        read_data_list = np.array([[1,0,1017],[2,0,543],[3,0,85],[4,0,92]])
        # 1017 is searching process's training image number in class 1, and so on
        npy_dir = dataset_path + '/valid_search/class'
    else:
        read_data_list = np.array([[1,0,2445],[2,0,1588],[3,0,401],[4,0,369]])
        # 2445 is rebuilding process's training image number in class 1, and so on
        npy_dir = dataset_path + '/train/class'
    while True:
        permutation = np.random.permutation(4) # randomly shuffle the order of 4 class images
        for i in range(4):
            read_list = read_data_list[i]
            batch_image_numpy[permutation[i]], batch_label_numpy[permutation[i]]\
            = data_aug(np.load(npy_dir+str(read_list[0])+'_image/'+str(read_list[1])+'.npy'),
            np.load(npy_dir+str(read_list[0])+'_label/'+str(read_list[1])+'.npy').astype(np.float32),
            aug_name, aug_proportion)
        read_data_list[:,1] = (read_data_list[:,1]+1)%read_data_list[:,2]
        yield batch_image_numpy, batch_label_numpy


def valid_test(net, batch_size, dataset_path, fun_type = 'valid', rebuild = False, **params_dic):
    '''
    This is evaluating function for validation and testing
    Args:
        net: network structure
        fun_type: 'valid' or 'test'
        rebuild: if True, this is evaluation process in rebuilding process, if False, this is evaluation process in searching process
        params_dic: the dictionary may include the following parameters
            highest_dice: needed for validation, the highest dice coefficient in the previous evaluation
            load_checkpoint: needed for testing, the path of the checkpoint
            dataset_path: dictionary path of the dataset
            epoch: needed for validation: the current epoch number
            save_file_name: needed for testing, the file name to save the final testing result
    '''
    net.eval()
    if fun_type == 'valid':
        assert 'highest_dice' in params_dic, 'Need to support highest_dice'
        image_number = 41
    else:
        assert 'load_checkpoint' in params_dic, 'Need to support load_checkpoint'
        net.load_state_dict(torch.load(params_dic['load_checkpoint']))
        image_number = 48
    total_areas = np.zeros([4, 2]).astype(np.float32)
    acc = 0
    for number_image in range(image_number):
        unit_image = np.load(dataset_path+'/'+fun_type+'_image/'+str(number_image)+'.npy')
        unit_label = np.load(dataset_path+'/'+fun_type+'_label/'+str(number_image)+'.npy').astype(np.float32)
        unit_image = crop_image(unit_image)
        unit_image = np.transpose(unit_image, (0, 3, 1, 2))
        test_total_batch = get_total_batch(batch_size, unit_image.shape[0])
        predict = np.zeros((unit_image.shape[0],unit_label.shape[2],unit_image.shape[2],unit_image.shape[3])).astype(np.float32)
        unit_image = torch.from_numpy(unit_image).float().cuda()
        all_batch_image = get_batch(unit_image, batch_size)	
        for now_batch in range(test_total_batch):
            batch_image = all_batch_image()
            if rebuild:
                logits = net(batch_image)
            else:
                logits, weights_SAM, weights_CAM = net(batch_image)
            logits = (logits > 0.5).float()
            logits = logits.cpu().numpy()
            predict[now_batch*batch_size:(now_batch+1)*batch_size] = logits
        predict = np.transpose(predict, (0, 2, 3, 1))#???unknown influence
        predict = recover_image(predict, unit_label.shape[0], unit_label.shape[1])
        four_areas = get_seperate_areas(predict, unit_label)
        total_areas += four_areas
        acc += np.sum((predict == unit_label).astype(np.float32)) / (predict.shape[0]*predict.shape[1]*predict.shape[2])
    total_dice = calculate_dice(total_areas)
    total_acc = acc / image_number
    average_dice = np.mean(total_dice)

    print('average_dice: {}, accuracy: {}'.format(average_dice, total_acc))
    for number_four_dice in range(4):
        print('dice {:.0f} is {:.4f}'.format(number_four_dice, total_dice[number_four_dice]))#
	
    if fun_type == 'valid':
        assert 'epoch' in params_dic,'Need to support epoch number' 
        if average_dice > params_dic['highest_dice'] and (rebuild or params_dic['epoch'] > 60):
            params_dic['highest_dice'] = average_dice
            print('xxxxxxxxxxxxxxxxx highest_dice: ',params_dic['highest_dice'],' saved! xxxxxxxxxxxxxxxxxxx')
            if rebuild:
                pth_name = 'RebuildSave.pth'
            else:
                pth_name = 'SearchSave.pth'
            torch.save(net.state_dict(),pth_name)
        return params_dic['highest_dice']
    elif fun_type == 'test':
        assert 'save_file_name' in params_dic, 'Need to support results saving file name'
        if not os.path.exists('results'):
            os.mkdir('results')
        f = open('results/'+params_dic['save_file_name'], 'a')
        f.write('Average_dice: {}, accuracy: {}\n'.format(average_dice, total_acc))
        for number_four_dice in range(4):
            f.write('dice {:.0f} is {:.4f}\n'.format(number_four_dice, total_dice[number_four_dice]))
        f.write('\n\n')
        f.close()
        if rebuild:
            return total_dice, average_dice, total_acc
        else:
            return weights_SAM, weights_CAM


def train_net(net, dataset_path, epochs = 100, batch_size = 12, lr = 1e-4, gpu = True, rebuild = False):
    '''
    This is training function for searching and rebuilding
    Args:
        dataset_path: dictionary path of the dataset
        epochs: total training epoch
        lr: learning rate
        gpu: if True, use CUDA to train, if False, not use CUDA
        rebuild: if False, do searching process training, if False, do rebuilding process training
    '''
    print('''
        Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        CUDA: {}
        '''.format(epochs, batch_size, lr, str(gpu)))

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = log_loss()
    highest_dice = 0
	
    def judge_grad(epoch, original_lr = lr):
        if epoch < 20 or epoch % 3 == 0:
            train_total_batch = get_total_batch(2, 2445)
            return False, True, original_lr, train_total_batch, False
            # architecture grad, network parameter grad, learning rate, total training batch, if search process
        else:
            train_total_batch = get_total_batch(2, 1017)
            return True, False, 1e-3, train_total_batch, True
    train_dataset = yield_data(dataset_path)
    if not rebuild:
        search_dataset = yield_data(dataset_path, search = True)
    for epoch in range(epochs):
        if epoch % 3 == 0:
            highest_dice = valid_test(net, batch_size, dataset_path, epoch = epoch, fun_type = 'valid', 
                        highest_dice = highest_dice, rebuild = rebuild)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        if rebuild:
            train_total_batch = get_total_batch(2, 2445)
        else:
            grad1, grad2, search_lr, train_total_batch, search = judge_grad(epoch)
            for k,v in net.named_parameters():
                if k == 'SAM_alphas' or k == 'CAM_alphas':
                    v.requires_grad = grad1
                else:
                    v.requires_grad = grad2		
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=search_lr)
        epoch_loss = torch.tensor([0]).float().cuda()
        for number_image in range(train_total_batch):	
            if rebuild:
                batch_image, batch_label = next(train_dataset)
            else:
                if search:
                    batch_image, batch_label = next(search_dataset)
                else:
                    batch_image, batch_label = next(train_dataset)
            batch_image = np.transpose(batch_image, (0, 3, 1, 2))
            batch_label = np.transpose(batch_label, (0, 3, 1, 2))	
            batch_image = torch.from_numpy(batch_image).float().cuda()
            batch_label = torch.from_numpy(batch_label).float().cuda()
            if rebuild:
                logits = net(batch_image)
            else:
                logits, weights_SAM, weights_CAM = net(batch_image)
                #print(weights_SAM)
            loss = criterion(logits, batch_label)
			
            epoch_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch finished ! Loss: {}'.format((epoch_loss / train_total_batch).cpu().detach().numpy()[0]))



def test_net(net, dataset_path, save_file_name, rebuild = False, batch_size = 12, gpu = True, pth_name = 'SearchSave.pth'):
    print('''
        Starting testing:
        Batch size: {}
        CUDA: {}
        Checkpoint name: {}
        '''.format(batch_size, str(gpu), pth_name))
    return valid_test(net, batch_size, dataset_path, fun_type = 'test', load_checkpoint = pth_name, save_file_name = save_file_name, rebuild = rebuild)







