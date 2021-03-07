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
import cv2 as cv
from utils.bbox_transform import bbox_overlaps_batch

def plot_detecton_boxes(image, debug_info, dets, class_name):
    image_info = debug_info['image_info'][0].cpu().numpy()
    real_gt_boxes = debug_info['real_gt_boxes']
    num_real_gt_boxes = int(image_info[4])

    overlaps = bbox_overlaps_batch(dets[:, :4], real_gt_boxes[:, :4]).squeeze(0)
    box_label = (overlaps >= 0.5).sum(dim=1)
    dets = dets.cpu().numpy()
    for i in range(np.minimum(100, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score<0.3:
            continue
        if box_label[i] == 0:
            cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 0, 255), 2)
        else:
            cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 255, 0), 2)
        text_width, text_height = \
        cv.getTextSize('{:s}: {:.3f}'.format(class_name, score), cv.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        box_coords = ((bbox[0], bbox[1] + 15), (bbox[0] + text_width + 2, bbox[1] + 15 - text_height - 2))
        cv.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED)
        cv.putText(image, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    for i in range(num_real_gt_boxes):
        bbox = tuple(int(np.round(x)) for x in real_gt_boxes[i, :4].cpu())
        cv.rectangle(image, bbox[0:2], bbox[2:4], (0, 255, 255), 3)
    return image

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
    watch_list = []
    debug_dir = os.path.join(cfg.DATA_DIR, 'debug', 'session_' + str(session))
    save_det_dir = os.path.join(debug_dir, 'detection_boxes')
    if not os.path.exists(save_det_dir):
        os.makedirs(save_det_dir)

    for i, data in enumerate(loader):
        image_data = data[0].to(device)
        image_info = data[1].to(device)
        ss_boxes = data[2].to(device)
        image_labels = data[3]
        image_ids = data[4]
        real_gt_boxes = data[5].to(device)

        det_tic = time.time()
        with torch.no_grad():
            combined_scores = wsddn(image_data, image_info, ss_boxes).squeeze(0)
        ss_boxes /= image_info[0][2].item()

        if i%300==0:
            watch_list.append(image_ids[0])
            debug_info = {}
            image = cv.imread(dataset.image_path_at(image_ids[0]))
            debug_info['real_gt_boxes'] = real_gt_boxes[0] / image_info[0][2].item()
            debug_info['image_info'] = image_info

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
                if image_ids[0] in watch_list:
                    image = plot_detecton_boxes(image, debug_info, cls_dets, dataset.classes[j])
            else:
                all_boxes[j][i] = torch.empty(0, 5).numpy()

        if image_ids[0] in watch_list:
            save_det_path = os.path.join(save_det_dir, image_ids[0] + '_' + '_epoch_' + str(epoch) + '_det.jpg')
            cv.imwrite(save_det_path, image)

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
