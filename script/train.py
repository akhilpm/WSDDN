import os
import time
import torch
import numpy as np
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from dataset.collate import collate_train
from model.wsddn import WSDDN
from utils.net_utils import clip_gradient
from sklearn.preprocessing import MultiLabelBinarizer

def train(dataset, net, batch_size, learning_rate, optimizer, lr_decay_step,
          lr_decay_gamma, pretrain, resume,  total_epoch,
          display_interval, session, epoch, save_dir, mGPU, log, add_params):
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    print(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    if batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = batch_size
    if learning_rate is not None:
        cfg.TRAIN.LEARNING_RATE = learning_rate
    if lr_decay_step is not None:
        cfg.TRAIN.LR_DECAY_STEP = lr_decay_step
    if lr_decay_gamma is not None:
        cfg.TRAIN.LR_DECAY_GAMMA = lr_decay_gamma

    if 'cfg_file' in add_params:
        update_config_from_file(add_params['cfg_file'])

    log.info(Back.WHITE + Fore.BLACK + 'Using config:')
    log.info('GENERAL:')
    log.info(cfg.GENERAL)
    log.info('TRAIN:')
    log.info(cfg.TRAIN)

    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params)
    loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True, collate_fn=collate_train)

    if 'data_path' in add_params:
        cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.join(cfg.DATA_DIR, save_dir, net, ds_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(Back.CYAN + Fore.BLACK + 'Output directory: %s' % (output_dir))

    if net == 'vgg16':
        wsddn = WSDDN(dataset.num_classes)
    elif net.startswith('resnet'):
        num_layers = net[6:]
        wsddn = WSDDN(dataset.num_classes, num_layers)
    else:
        raise ValueError(Back.RED + 'Network "{}" is not defined!'.format(net))

    wsddn.to(device)

    params = []
    for key, value in dict(wsddn.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value],
                            'lr': cfg.TRAIN.LEARNING_RATE * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value],
                            'lr':cfg.TRAIN.LEARNING_RATE,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if optimizer == 'sgd':
        optimizer = SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    elif optimizer == 'adam':
        optimizer = Adam(params)
    else:
        raise ValueError(Back.RED + 'Optimizer "{}" is not defined!'.format(optimizer))

    start_epoch = 1
    data_size = len(dataset)

    if pretrain or resume:
        model_name = 'wsddn_{}_{}.pth'.format(session, epoch)
        if 'model_name' in add_params:
            model_name = '{}.pth'.format(add_params['model_name'])
        model_path = os.path.join(output_dir, model_name)
        print(Back.WHITE + Fore.BLACK + 'Loading checkpoint %s...' % (model_path))
        checkpoint = torch.load(model_path, map_location=device)
        wsddn.load_state_dict(checkpoint['model'])
        if resume:
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        print('Done.')

    # Decays the learning rate of each parameter group by gamma every step_size epochs.
    lr_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.LR_DECAY_GAMMA)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit([np.arange(1, 21)])

    if mGPU:
        wsddn = torch.nn.DataParallel(wsddn)
    wsddn.train()

    for current_epoch in range(start_epoch, total_epoch + 1):
        loss_temp = 0
        start = time.time()
        total_loss = 0.0

        for step, data in enumerate(loader):
            image_data = data[0].to(device)
            image_info = data[1].to(device)
            ss_boxes = data[2].to(device)
            image_labels = data[3]
            image_ids = data[4]
            binary_targets = label_binarizer.transform(image_labels)
            binary_targets = torch.from_numpy(binary_targets.astype(np.float32)).to(device)

            combined_scores = wsddn(image_data, image_info, ss_boxes)
            loss = wsddn.calculate_loss(combined_scores, binary_targets)
            loss_temp += loss.item()
            total_loss += loss.item() * cfg.TRAIN.BATCH_SIZE

            optimizer.zero_grad()
            loss.backward()
            if net == 'vgg16':
                clip_gradient(wsddn, 10.)
            optimizer.step()

            if step % display_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (display_interval + 1)

                print(Back.WHITE + Fore.BLACK + '[session %d][epoch %2d/%2d][iter %4d/%4d]'
                      % (session, current_epoch, total_epoch, step, len(loader)))
                print('loss: %.4f, learning rate: %.2e, time cost: %f'
                      % (loss_temp, optimizer.param_groups[0]['lr'], end-start))
                loss_temp = 0
                start = time.time()
        total_loss = total_loss / data_size
        log.info("Epoch: {} Loss: {:.3f}".format(current_epoch, total_loss))
        lr_scheduler.step()

        save_path = os.path.join(output_dir, 'wsddn_{}_{}.pth'.format(session, current_epoch))
        checkpoint = {'epoch': current_epoch + 1,
                      'model': wsddn.module().state_dict() if mGPU else wsddn.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, save_path)
        print(Back.WHITE + Fore.BLACK + 'Model saved: %s' % (save_path))

    print(Back.GREEN + Fore.BLACK + 'Train finished.')
