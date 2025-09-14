# unit test

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import boundary_selection
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
from torch import nn
import torchvision.datasets as dset
import torch.nn.functional as F
import time
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn

import sys
sys.path.append('/bood/scripts')
from resnet_anchor import ResNet_Model 

num_classes = 100

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='cifar100')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='Batch size.')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Checkpoints
    parser.add_argument('--load', '-l', type=str, default='',
                        help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    # EG specific
    parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--save', type = str)
    parser.add_argument('--device', type = str)
    parser.add_argument('--ckpt', type = str)
    parser.add_argument('--rate', type = float)
    parser.add_argument("--arch", default="resnet34", help="backbone architecture")
    parser.add_argument('--step_num', type = int, default = 40)
    parser.add_argument('--datapath', type = str)
    parser.add_argument('--step_size', type = float, default = 0.01)
    parser.add_argument('--testacc', type=bool, default=False)
    parser.add_argument('--histo', action = 'store_true')
    parser.add_argument('--kappa', type=int)
    parser.add_argument('--eps', type=float, default=0.0003)
    parser.add_argument('--rand', action='store_true')
    args = parser.parse_args()
    return args

def GA_PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init, anchor):

    # model.eval()
    Kappa = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.0001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        # if rand then add a purturbation.
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x_adv), 1, 1),
                            x_adv.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1

        probabilities = F.softmax(output, dim=1)
        max_probabilities = torch.max(probabilities, dim=1).values.cpu().tolist()
        max_probabilities = [float(f"{number:.6f}") for number in max_probabilities]

        predict = torch.argmax(output, 1)

        # Update Kappa
        # if predict != target, cross the boundary.
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                Kappa[p] += 1
        
        
        with torch.enable_grad():
            if loss_fn == "cent":
                target = target.type(torch.LongTensor)
                target = target.to('cuda:0')
                loss_adv = F.cross_entropy(output, target)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() + eta

    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa

def get_kappa(args, model, data_loader, step_num, step_size, anchor):
    kappa_values = []
    targets = []
    
    for _, img, target in tqdm(data_loader):
        img, target = img.to(args.device), target.to(args.device)
        
        _, kappa = GA_PGD(model, img, target, epsilon=args.eps, 
                          step_size=step_size, num_steps=step_num, 
                          loss_fn="cent", category="Madry", rand_init=args.rand, anchor = anchor)

        kappa_values.extend(kappa.detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())
        
    return np.array(kappa_values), np.array(targets)

def get_prune_idx(args, kappa_values):
    
    sorted_idx = kappa_values.argsort()
    high_idx = round(kappa_values.shape[0] * args.rate)
    
    ids = sorted_idx[:high_idx]
    
    return ids


def get_idx(args, kappa_values):
    sorted_idx = kappa_values.argsort()
    high_idx = round(kappa_values.shape[0] * args.rate)
    
    ids_b = sorted_idx[:high_idx]
    ids_c = sorted_idx[high_idx:]
    
    return ids_b, ids_c


def get_prune_idx_group(args, kappa_values):
    ids_b = []
    ids_c = []
    if args.dataset == 'cifar100':
        num_classes = 100
        num_gp = 500
    elif args.dataset == 'in100':
        num_classes = 100
        num_gp = 1000
    kappa_group = [kappa_values[(num_gp * i):(num_gp * (i + 1))] for i in range(num_classes)]
    # kappa_group: num_classes, num_gp, 768
    for g in range(len(kappa_group)):
        _ = kappa_group[g].argsort()
        high_idx = round(num_gp * args.rate)
        ids_b.append(_[:high_idx])
        ids_c.append(_[high_idx:])

    return ids_b, ids_c

def get_specified_kappa_values(args, kappa_values):
    step = args.kappa
    indices = []
    for i, num in enumerate(kappa_values):
        if num == step:
            indices.append(i)
    return indices

def check_acc(trainloader, anchor, num_classes):
    # check clf
    accs = []
    print(len(trainloader))
    for _, data, target in tqdm(trainloader):

        # move the batch of input to GPU
        data, target = data.cuda(), target.cuda()
        x = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(data), 1, 1),
                            data.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1
        pred = x.argmax(1)
        acc = torch.eq(pred, target).float().mean().cpu().numpy()
        accs.append(acc)
    print('acc: ', np.mean(accs))
    


class IDFeaturesDataset(Dataset):
    def __init__(self, id_features, transform=None, target_transform=None, dataset = None):
        print('ID dataset: ', dataset)
        
        self.samples = []
        if dataset == 'cifar100':
            self.class_names = np.zeros(500*100) 
            id_features = id_features.reshape((100, 500, 768))
            ordered = [i for i in range(100)]
            for i in range(100):
                self.class_names[i * 500:(i+1)*500] = ordered[i]
                # normalize the feature for each class
                ID = F.normalize(id_features[i], p=2, dim=1).cpu().numpy()
                self.samples.append(ID)
                
        elif dataset == 'in100':
            self.class_names = np.zeros(1000*100) 
            ordered = [i for i in range(100)]
            id_features = id_features.reshape((100, 1000, 768))
            for i in range(100):
                self.class_names[i * 1000:(i+1)*1000] = ordered[i]
                # normalize the feature for each class
                ID = F.normalize(id_features[i], p=2, dim=1).cpu().numpy()
                self.samples.append(ID)
        
        self.samples = np.array(self.samples)
                
        self.transform = transform
        self.target_transform = target_transform

        if dataset == 'in100':
            self.samples = self.samples.reshape(100000, 768)
        else:
            self.samples = self.samples.reshape(50000, 768)

        print(self.class_names.shape)
        print(self.samples.shape)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        feature, target = self.samples[idx], self.class_names[idx]

        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return idx, feature, target

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def main():

    args = parse_args()
    device = 'cuda'
    cudnn.benchmark = True  
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    
    os.makedirs(args.save, exist_ok=True)
    
    torch.manual_seed(1)
    np.random.seed(args.seed)
    
    
    if args.dataset == 'cifar100':
        num_classes = 100
        anchor = torch.from_numpy(np.load('./token_embed_c100.npy')).cuda()
        data_path = args.datapath
    if args.dataset == 'in100':
        num_classes = 100
        anchor = torch.from_numpy(np.load('./token_embed_in100.npy')).cuda()
        data_path = args.datapath
    
    id_features = torch.from_numpy(np.load(data_path, allow_pickle=True)).cuda()
    data_train = IDFeaturesDataset(id_features, dataset=args.dataset)

    trainloader = torch.utils.data.DataLoader(
        data_train,
        batch_size=64, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)
    net = ResNet_Model(name='resnet34', num_classes=num_classes)
    if args.dataset == 'in100':
        net.load_state_dict(remove_data_parallel(torch.load(args.ckpt)))
    else:
        net.load_state_dict(torch.load(args.ckpt))
    
    net.cuda()
    torch.cuda.manual_seed(1)
    if args.testacc:
        check_acc(trainloader, anchor, num_classes = num_classes)
    
    kappa_values, targets = get_kappa(args, net, trainloader, step_num = args.step_num, step_size = args.step_size, anchor = anchor)
    ids_boundary, ids_core = get_prune_idx_group(args, kappa_values)
    kappa_idx = get_specified_kappa_values(args, kappa_values)
    print(f'number of features with kappa value = {args.kappa} ', len(kappa_idx))
    
    save_boundary = os.path.join(args.save, "boundary.bin")
    save_core = os.path.join(args.save, "core.bin")
    save_kappa = os.path.join(args.save, "kappa.bin")
    save_kappa_step = os.path.join(args.save, "kappa_step.bin")
    
    with open(save_boundary, "wb") as file:
        pickle.dump(ids_boundary, file)
    with open(save_core, "wb") as file:
        pickle.dump(ids_core, file)
    with open(save_kappa, "wb") as file:
        pickle.dump(kappa_values, file)
    with open(save_kappa_step, "wb") as file:
        pickle.dump(kappa_idx, file)


    
if __name__ == "__main__":
    main()
    
    