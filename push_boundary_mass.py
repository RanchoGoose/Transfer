import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import sys
sys.path.append('/bood/scripts')
from  resnet_anchor import ResNet_Model
from utils import get_model

def GA_PGD_1(model, data, target, epsilon, 
             step_size, num_steps,loss_fn,
             category,rand_init, step_outside, anchor, 
             num_classes, total_step, thre, 
             max_feat, method, cross, cstep, ori, filt):

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(
            np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
    oods, ood_target = [], []
    crossed = [] 
    output0 = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x_adv), 1, 1), 
                                     x_adv.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1
    
    probabilities = F.softmax(output0, dim=1)
    max_probabilities = torch.max(probabilities, dim=1).values.cpu().tolist()
    max_probabilities = [float(f"{number:.6f}") for number in max_probabilities]
    predict = torch.argmax(output0, 1)
    print()
    print(predict)
    print(max_probabilities)

    filt = []

    if ori:
        for p in range(len(x_adv)):
            oods.append(x_adv[p].detach())
            ood_target.append(target[p])
        return x_adv, oods, ood_target
    
    if cross:
        steps = [-1 for _ in range(len(x_adv))]
        added = []
        for k in range(total_step + 1):
            x_adv.requires_grad_()
            output = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x_adv), 1, 1), 
                                         x_adv.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1
            probabilities = F.softmax(output, dim=1)
            max_probabilities = torch.max(probabilities, dim=1).values.cpu().tolist()
            predict = torch.argmax(output, 1)
            max_probabilities = [float(f"{number:.6f}") for number in max_probabilities]
            for p in range(len(x_adv)):
                if method == 'threashold_filtering':
                    if max_probabilities[p] <= thre:
                        oods.append(x_adv[p].detach())
                        ood_target.append(target[p])
                        # print(f'feature {p} was appended in step {k}')
                if method == 'boundary_crossing':
                    if predict[p] != target[p] and steps[p] != cstep:
                        steps[p] += 1
    
                    if predict[p] != target[p] and steps[p] == cstep and p not in added:
                        steps[p] += 1
                        oods.append(x_adv[p].detach())
                        ood_target.append(target[p])
                        added.append(p)
                        print(predict[p], target[p])
                        print(max_probabilities[p])
                        filt.append(max_probabilities[p])
                        print(f'feature {p} was appended in step {k}')
                    if len(crossed) == len(x_adv):
                        break
            if len(ood_target) == max_feat:
                break
            # output.zero_grad()
            # model.zero_grad()
            with torch.enable_grad():
                if loss_fn == "cent":
                    target = target.type(torch.LongTensor)
                    target = target.to('cuda:0')
                    # loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
                    
                    loss_adv = F.cross_entropy(output, target)
                    # print(loss_adv)
                if loss_fn == "kl":
                    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                    loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            loss_adv.backward() 
            eta = step_size * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
    else:
        for k in range(total_step + 1):
            x_adv.requires_grad_()
            output = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(x_adv), 1, 1), 
                                         x_adv.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1
            probabilities = F.softmax(output, dim=1)
            max_probabilities = torch.max(probabilities, dim=1).values.cpu().tolist()
            predict = torch.argmax(output, 1)

            max_probabilities = [float(f"{number:.6f}") for number in max_probabilities]
            for p in range(len(x_adv)):
                if method == 'threashold_filtering':
                    if max_probabilities[p] <= thre:
                        oods.append(x_adv[p].detach())
                        ood_target.append(target[p])

                if method == 'boundary_crossing':
                    if predict[p] != target[p] and p not in crossed:
                        oods.append(x_adv[p].detach())
                        ood_target.append(target[p])
                        crossed.append(p)
                        print(predict[p], target[p])
                        print(max_probabilities[p])
                        print(f'feature {p} was appended in step {k}')

                    if len(crossed) == len(x_adv):
                        break
            if len(ood_target) == max_feat:
                break
            with torch.enable_grad():
                if loss_fn == "cent":
                    target = target.type(torch.LongTensor)
                    target = target.to('cuda:0')                    
                    loss_adv = F.cross_entropy(output, target)
                    # print(loss_adv)
                if loss_fn == "kl":
                    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                    loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            loss_adv.backward() # compute grad here
            eta = step_size * x_adv.grad.sign()
            x_adv = x_adv.detach() + eta
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, oods, ood_target, filt

def get_ood1(args, model, data_loader, anchor, num_classes, opt, i):
    ood_embeddings = None
    filt = None
    for _, img, target in tqdm(data_loader):
        # print(img.shape, target.shape)
        img, target = img.to(args.device), target.to(args.device)
        # print(img.shape, target.shape)
        # print(0)

        # get the kappa values from PGD
        x_adv, oods, ood_target,filt = GA_PGD_1(model, img, target, 
                                         epsilon=args.eps, num_steps=40, 
                                         loss_fn="cent", category="Madry", rand_init=args.rand, 
                                         step_outside=args.step_outside, anchor = anchor, 
                                         num_classes=num_classes, step_size = args.step_size,
                                         total_step=i, thre = args.thre, max_feat = args.max_feat, method = args.method, 
                                         cross = args.cross, cstep = args.cstep, ori = args.ori, filt = args.filt)
        ood_embeddings = oods

        print(len(oods))
    if args.filt:
        ood_embeddings = filt_oods(args, filt, ood_embeddings)
    
    ood_embeddings = torch.stack(ood_embeddings, dim = 0)
    return ood_embeddings, filt # return a torch tensor of ood embeddings


class SingleBoundaryFeaturesDataset(Dataset):
    def __init__(self, id_features, boundary_id, dataset, class_idx, transform=None, target_transform=None):
        # print('ID dataset: ', dataset)
        self.targets = np.zeros(500*100) 
        if dataset == 'cifar100':
            class_size = 100
            ordered = [i for i in range(100)]
            for i in range(100):
                self.targets[i * 500:(i+1)*500] = ordered[i]
        elif dataset == 'in100':
            self.targets = np.zeros(1000*100) 
            class_size = 100
            ordered = [i for i in range(100)]
            for i in range(100):
                self.targets[i * 1000:(i+1)*1000] = ordered[i]
        self.samples = np.array(id_features)
        self.samples = torch.from_numpy(self.samples).cuda()
        for i in range(self.samples.shape[0]):
            self.samples[i] = F.normalize(self.samples[i], p = 2, dim = 1)
        new_sample = []
        new_target = []
        if dataset == 'cifar100':

            self.targets = self.targets.reshape((100, 500))
            for i in range(100):
                sample = self.samples[i][boundary_id[i]].cpu().numpy()

                new_sample.append(sample)
                target = self.targets[i][boundary_id[i]]
                new_target.append(target)
        if dataset == 'in100':
            self.samples = self.samples.reshape((100, 1000, 768))
            self.targets = self.targets.reshape((100, 1000))
            for i in range(100):
                sample = self.samples[i][boundary_id[i]].cpu().numpy()
                # print(len(sample))
                new_sample.append(sample)
                target = self.targets[i][boundary_id[i]]
                new_target.append(target)
        
        # 1000, 768
        self.samples = np.vstack(new_sample)
        self.targets = np.hstack(new_target)

        class_num = int(self.samples.shape[0] / class_size)
        self.samples = self.samples[class_idx * class_num : (class_idx + 1) * class_num] 
        self.targets = self.targets[class_idx * class_num : (class_idx + 1) * class_num]
        
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        feature, target = self.samples[idx], self.targets[idx]

        return idx, feature, target


def check_acc(trainloader, anchor, num_classes):
    # check clf
    accs = []
    for _, data, target in tqdm(trainloader):

        # move the batch of input to GPU
        data, target = data.cuda(), target.cuda()
        x = F.cosine_similarity(anchor.unsqueeze(0).repeat(len(data), 1, 1),
                            data.unsqueeze(1).repeat(1, num_classes, 1), 2) / 0.1
        pred = x.argmax(1)
        acc = torch.eq(pred, target).float().mean().cpu().numpy()
        accs.append(acc)
    print('acc: ', np.mean(accs))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, help='Choose dataset')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    parser.add_argument('--gaussian', type = float, default=0)
    
    parser.add_argument('--save', type = str)
    parser.add_argument('--device', type = str, default='cuda')
    parser.add_argument('--ckpt', type = str)
    parser.add_argument("--arch", default="resnet34", help="backbone architecture")
    parser.add_argument('--testacc', action='store_true')
    parser.add_argument('--step_num', type = int, default = 40)
    parser.add_argument('--datapath', type = str)
    parser.add_argument('--boundary_id', '-bid', type=str, help = 'path to boundary id')
    parser.add_argument('--id_kappa', type=str)
    parser.add_argument('--thre', type = float, default = 0.2, help = 'threshold of considering a feature is an OOD feature.')
    parser.add_argument('--max_feat', type = int, default = 10000)
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--step_size', type = float, default = 0.01)
    parser.add_argument('--step_outside', '-so', type = int, default=1, help='number of steps pushing outside.')
    # parser.add_argument('--kappa', type = int)
    parser.add_argument('--step_feat', action='store_true')
    parser.add_argument('--total_step', type = int)
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--rand', action='store_true')
    # parser.add_argument('--net', action='store_true')
    parser.add_argument('--method', type = str)
    parser.add_argument('--cross', action = 'store_true', help= 'add the step after cross the boundary')
    parser.add_argument('--rate',type = float, help = 'rate for filtering boundary in the previous step, if using boundary crossing, then should be None.')
    parser.add_argument('--cstep', type = int, default = 1)
    parser.add_argument('--ori', action = 'store_true')
    parser.add_argument('--filt', action = 'store_true')
    args = parser.parse_args()
    return args

def save_feat(args, num_classes, ood, anchor, step):
    ood = torch.stack(ood, dim =0)
    print(ood.shape)
    for i in range(num_classes):
        IDS = ood[i]
        
        idx = i # class idx
        # IDS = torch.stack(IDS, dim=0)
        IDS = IDS * anchor[idx].norm()
        ood[i] = IDS
        
    # print(len(ood))
    # print(len(IDS))
    
    save_ood = os.path.join(args.save, f"{args.dataset}_ood_{args.method}b_rate{args.rate}_stepsize{args.step_size}_step{step}_ori{args.ori}_cstep_{args.cstep}.npy")
    np.save(save_ood, ood.cpu().data.numpy())
    print(f'your boundary features have been stored in {save_ood}')

def filt_oods(args, filt, ood):
    paired = list(zip(filt, ood))
    
    # Sort the pairs based on probabilities (ascending order)
    sorted_pairs = sorted(paired, key=lambda x: x[0].item() if isinstance(x[0], torch.Tensor) else x[0])
    
    # Unzip the sorted pairs
    filt, ood = zip(*sorted_pairs)
    ood = ood[:int(len(ood)/2)]
    return  list(ood)

from collections import OrderedDict
def remove_data_parallel(old_state_dict):
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict
    
def main():

    args = parse_args()

    if os.path.isdir(args.save) == False:
        os.mkdir(args.save)
    
    device = 'cuda'
    cudnn.benchmark = True  
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    
    torch.manual_seed(1)
    np.random.seed(args.seed)
    
    if args.dataset == 'cifar100':
        num_classes = 100
        anchor = torch.from_numpy(np.load('./token_embed_c100.npy')).cuda()
        data_path = args.datapath
    elif args.dataset == 'in100':
        num_classes = 100
        anchor = torch.from_numpy(np.load('./token_embed_in100.npy')).cuda()
        data_path = args.datapath
    
    id_features = np.load(data_path, allow_pickle=True)
    print(id_features.shape)
    id_features = [np.array(feature) for feature in id_features]


    ids = args.boundary_id
    with open(ids, "rb") as f:
        idx_boundary = pickle.load(f)
    # create the model
    net = ResNet_Model(name='resnet34', num_classes=num_classes)
    
    if args.dataset == 'in100':
        net.load_state_dict(remove_data_parallel(torch.load(args.ckpt)))
    else:
        net.load_state_dict(torch.load(args.ckpt))
    net.cuda()
    torch.cuda.manual_seed(1)
    
    
    opt = torch.optim.SGD(
        list(net.parameters()),
        0.1, momentum=0.9,
        weight_decay=0.0005, nesterov=True)
    print(args.method)
    oods = []
    for i in range(num_classes):
        data_train = SingleBoundaryFeaturesDataset(id_features, idx_boundary, args.dataset, i)
        trainloader = torch.utils.data.DataLoader(
            data_train,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.prefetch, pin_memory=True)

        # get_ood1(args, model, data_loader, anchor, num_classes, opt, i)
        if args.method == 'boundary_crossing':
            ood, filt = get_ood1(args, net, trainloader, anchor, num_classes, opt, args.total_step)
        

        oods.append(ood)


    # print(len(oods))
    # oods = torch.stack(oods, 0)
    save_feat(args, num_classes, oods, anchor, args.total_step)
    
    
if __name__ == "__main__":
    main()
