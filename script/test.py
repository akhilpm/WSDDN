import os
import sys
import time
import pickle
import torch
import numpy as np
import dataset.dataset_factory as dataset_factory
from colorama import Back, Fore
from config import cfg, update_config_from_file
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from model.wsddn import WSDDN
from torchvision.ops import nms

def test(dataset, net, load_dir, session, epoch, log, add_params):
    log.info("============== Testing EPOCH {} =============".format(epoch))
    device = torch.device('cuda:0') if cfg.CUDA else torch.device('cpu')
    log.info(Back.CYAN + Fore.BLACK + 'Current device: %s' % (str(device).upper()))

    if 'cfg_file' in add_params:
        update_config_from_file(add_params['cfg_file'])

    log.info(Back.WHITE + Fore.BLACK + 'Using config:')
    log.info('GENERAL:')
    log.info(cfg.GENERAL)
    log.info('TEST:')
    log.info(cfg.TEST)

    # TODO: add competition mode
    dataset, ds_name = dataset_factory.get_dataset(dataset, add_params, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_test)

    if 'data_path' in add_params: cfg.DATA_DIR = add_params['data_path']
    output_dir = os.path.join(cfg.DATA_DIR, 'output', net, ds_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(Back.CYAN + Fore.BLACK + 'Output directory: %s' % (output_dir))

    if net == 'vgg16':
        wsddn = WSDDN(dataset.num_classes)
    elif net.startswith('resnet'):
        num_layers = net[6:]
        wsddn = WSDDN(dataset.num_classes, num_layers)
    else:
        raise ValueError(Back.RED + 'Network "{}" is not defined!'.format(net))

    wsddn.to(device)

    model_path = os.path.join(cfg.DATA_DIR, load_dir, net, ds_name, 
                              'wsddn_{}_{}.pth'.format(session, epoch))
    log.info(Back.WHITE + Fore.BLACK + 'Loading model from %s' % (model_path))
    checkpoint = torch.load(model_path, map_location=device)
    wsddn.load_state_dict(checkpoint['model'])
    log.info('Done.')

    start = time.time()
    max_per_image = 100
        
    all_boxes = [[[] for _ in range(len(dataset))] for _ in range(dataset.num_classes)]

    wsddn.eval()

    for i, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)
        ss_boxes = data[2].to(device)

        det_tic = time.time()
        with torch.no_grad():
            combined_scores = wsddn(image_data, image_info, ss_boxes).squeeze(0)

        ss_boxes /= image_info[0][2].item()
        det_toc = time.time()
        detect_time = det_toc - det_tic

        misc_tic = time.time()
        for j in range(dataset.num_classes-1):
            inds = torch.nonzero(combined_scores[:,j] > 0.0).view(-1)
            if inds.numel() > 0:
                selected_scores = combined_scores[:, j][inds]
                selected_boxes = ss_boxes[0, inds]
                cls_dets = torch.cat((selected_boxes, selected_scores.unsqueeze(1)), 1)
                keep = nms(selected_boxes, selected_scores, cfg.TEST.NMS)
                all_boxes[j][i] = cls_dets[keep.view(-1).long()].cpu().numpy()
            else:
                all_boxes[j][i] = torch.empty(0, 5).numpy()

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, len(dataset), detect_time, nms_time))
        sys.stdout.flush()
            
    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    log.info('\nEvaluating detections...')
    dataset.evaluate_detections(all_boxes, output_dir, log)

    # TODO: Add txt file with result info ?

    end = time.time()
    log.info(Back.GREEN + Fore.BLACK + 'Test time: %.4fs.' % (end - start))
