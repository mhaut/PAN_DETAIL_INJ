# -*- coding: utf-8 -*-
from h5py import File as load
import numpy as np
import mydataset
from torch.utils.data import DataLoader


def get_data(data_file, label_file):
    data  = load('{}'.format(data_file), 'r')
    label = load('{}'.format(label_file), 'r')
    data, label = data['data'], label['label']
    data  = np.transpose(data, [0, 1, 3, 2])
    label = np.transpose(label, [0, 1, 3, 2])
    assert len(data) == len(label)
    return data, label

def load_data(dataset, bsizes):
    pathfile = './data/'+dataset+'/size_33/'
    tr_data, tr_labels   = get_data(pathfile + 'train_data.mat',    pathfile + 'train_label.mat')
    val_data, val_labels = get_data(pathfile + 'validate_data.mat', pathfile + 'validate_label.mat')
    te_data, te_labels   = get_data(pathfile + 'test_data.mat',     pathfile + 'test_label.mat')
    train_hyper = mydataset.MyDataset((tr_data, tr_labels))
    val_hyper   = mydataset.MyDataset((val_data, val_labels))
    te_hyper    = mydataset.MyDataset((te_data, te_labels))
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = DataLoader(train_hyper, batch_size=bsizes[0], shuffle=True, drop_last=True, **kwargs)
    val_loader   = DataLoader(val_hyper, batch_size=bsizes[1], shuffle=False, **kwargs)
    te_loader    = DataLoader(te_hyper, batch_size=bsizes[1], shuffle=False, **kwargs)
    return train_loader, val_loader, te_loader, tr_data.shape[1]
