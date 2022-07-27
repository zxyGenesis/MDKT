import torch
from torch.autograd import Variable
import myMetaDataset
import ResNet
import yaml
import data
import os
import argparse
import numpy as np
import h5py
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def save_features(model, data_loader, outfile ):

    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', (max_count, feats.size(1)), dtype='f')
        all_feats[count:count+feats.size(0),:] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Save features')
    parser.add_argument('--cfg', required=True, help='yaml file containing config for data')
    parser.add_argument('--outfile', required=True, help='save file')
    return parser.parse_args()
    # python save_features.py --cfg 'train_save_data.yaml' --outfile 'train.hdf5'
    # python save_features.py --cfg 'test_save_data.yaml' --outfile 'test.hdf5'

if __name__ == '__main__':
    params = parse_args()
    with open(params.cfg,'r') as f:
        data_params = yaml.load(f)

    data_loader = data.get_data_loader(data_params)
    model = ResNet.resnet50()

    tmp = torch.load('BackboneModel/Resnet-50.pth')
    model.load_state_dict(tmp)

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    save_features(model, data_loader, params.outfile)
