import os
import json
import argparse
from transformers import logging
logging.set_verbosity_error()
import logging
import torch
from util.logger import set_logger
from util.data import generate_loader
from torch import nn

from segformer import SegformerForSemanticSegmentation, SegformerConfig
import time
import importlib
from tqdm.auto import tqdm
from util.utils import label_accuracy_score, add_hist
import numpy as np
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

import torch.nn.functional as F


def train(model, train_loader, optimizer, scheduler, num_labels, dev=None):
    model.train()
    logging.info('Training')
    train_running_loss = 0.0
    counter = 0
    hist = np.zeros((num_labels, num_labels))
    for step, inputs in tqdm(enumerate(train_loader)):
        counter += 1
        imgs = inputs['pixel_values'].to(dev)
        labels = inputs['labels'].to(dev, dtype=torch.long)
        # Forward pass
        outputs = model(pixel_values=imgs, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']
        # Calculate the loss
        loss = loss.mean()
        train_running_loss += loss.item()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        preds = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        hist = add_hist(hist, labels, preds, n_class=num_labels)

    epoch_loss = train_running_loss / counter
    _, _, epoch_miou, _, _ = label_accuracy_score(hist)
    epoch_lr = optimizer.param_groups[0]["lr"]
    return epoch_loss, epoch_miou, epoch_lr


def validate(model, val_loader, num_labels, category_names, dev=None):
    model.eval()
    logging.info('Validation')
    valid_running_loss = 0.0
    counter = 0
    hist = np.zeros((num_labels, num_labels))
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(val_loader)):
            counter += 1
            imgs = inputs['pixel_values'].to(dev)
            labels = inputs['labels'].to(dev, dtype=torch.long)
            # Forward pass
            outputs = model(pixel_values=imgs, labels=labels)
            # Calculate the loss
            logits, loss = outputs['logits'], outputs['loss']
            preds = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            valid_running_loss += loss.item()
            # Calculate the accuracy
            hist = add_hist(hist, labels, preds, n_class=num_labels)
    # Loss and accuracy for the complete epoch
    _, _, epoch_miou, _, IoU = label_accuracy_score(hist)
    epoch_miou_by_class = [{classes : round(IoU, 4)} for IoU, classes in zip(IoU, category_names)]
    epoch_loss = valid_running_loss / counter
    return epoch_loss, epoch_miou, epoch_miou_by_class


def main(opt):
    
    # logging.info(f"Training config: \n{opt}")
    
    torch.manual_seed(opt.seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(opt.data_dir, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    ### training from scratch 
    if opt.from_scratch:
        model = SegformerForSemanticSegmentation(
            SegformerConfig(
                num_labels=len(id2label), 
                id2label=id2label, 
                label2id=label2id, 
                ignore_mismatched_sizes=True)
        )
        
    ### fine-tuning
    else:
        # dir version
        if os.path.isdir(opt.pretrain)==True:
            model = SegformerForSemanticSegmentation.from_pretrained(
                opt.pretrain, 
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        # .pth file version
        else:
            model = SegformerForSemanticSegmentation.from_pretrained(
                None,
                opt.pretrain, 
                SegformerConfig(
                    num_labels=len(id2label), 
                    id2label=id2label, 
                    label2id=label2id, 
                    ignore_mismatched_sizes=True)
            )
    
    model = model.to(dev)
    model = nn.DataParallel(model).to(dev)
    params = []
    
    ### training from scratch
    if opt.from_scratch:
        for _, param in model.named_parameters(recurse=True):
            lr = opt.lr * 10
            decay = opt.weight_decay
            params.append({'params': param, 'lr': lr, 'weight_decay': decay})

    ### fine-tuning
    else:
        for layer, param in model.named_parameters(recurse=True):
            lr = opt.lr
            decay = opt.weight_decay
            if 'norm' in layer:
                decay = 0.0
            if 'decode' in layer:
                lr = opt.lr * 10.0
            params.append({'params': param, 'lr': lr, 'weight_decay': decay})


    optimizer = torch.optim.AdamW(
        params,
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    epochs = opt.epochs
    
    train_loader = generate_loader(opt, 'train')
    val_loader = generate_loader(opt, 'val')
    scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=opt.warmup_steps,
            num_training_steps=int(len(train_loader) * epochs),
            lr_end=0.0,
            power=1,
        )
    logging.info(f"Number of training images: {len(train_loader.dataset)}")
    logging.info(f"Number of validation images: {len(val_loader.dataset)}")
    logging.debug(f"Computation device: {dev}")
    logging.info(f"Epochs to train for: {epochs}\n")

    category_names = list(label2id.keys())
    num_labels = len(id2label)
    
    total_params = sum(list(map(lambda x: x.numel(), model.parameters())))
    logging.info(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"{total_trainable_params:,} training parameters.")
    
    best_val_miou = 0.0
    train_loss, val_loss = [], []
    train_miou, val_miou = [], []
    elapsed_time = []
    time_one_epoch_start = None
    time_one_epoch_end = None
    elapsed_time_one_epoch = None
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1} of {epochs}")
        time_one_epoch_start = time.time()
        train_epoch_loss, train_epoch_miou, train_epoch_lr = \
            train(model, train_loader, optimizer, scheduler, num_labels,dev)
        val_epoch_loss, val_epoch_miou, val_epoch_miou_by_class = \
            validate(model, val_loader, num_labels, category_names, dev)

        time_one_epoch_end = time.time()
        elapsed_time_one_epoch = int(time_one_epoch_end - time_one_epoch_start)
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_miou.append(train_epoch_miou)
        val_miou.append(val_epoch_miou)
        elapsed_time.append(elapsed_time_one_epoch)
        
        logging.info(f"Training loss: {train_epoch_loss:.3f} | Training miou: {train_epoch_miou:.3f} | Training lr: {train_epoch_lr}")
        logging.info(f"Validation loss: {val_epoch_loss:.3f} | Validation miou: {val_epoch_miou:.3f} | Validation miou by class: {val_epoch_miou_by_class}")
        logging.info('-'*50)
        
        if best_val_miou < val_epoch_miou:
            best_val_miou = val_epoch_miou
            model.module.save_pretrained(os.path.join(opt.save_path, 'best'))
        time.sleep(5)
        
    model.module.save_pretrained(os.path.join(opt.save_path, 'final'))
    logging.info('TRAINING COMPLETE!')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0,1', help='Select gpu to use')
    parser.add_argument('--lr', type=float, default=6e-5) # do not modify
    parser.add_argument('--pretrain', type=str, default='nvidia/mit-b2')
    parser.add_argument('--save_path', type=str, default='result/')
    parser.add_argument('--num_workers', type=int, default=4) # do not modify
    parser.add_argument('--seed', type=int, default=1) # do not modify
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--epochs', type=int, default=120) # do not modify
    parser.add_argument('--from_scratch', type=bool, default=True)
    parser.add_argument('--warmup_steps', type=int, default=1500) # do not modify
    parser.add_argument('--weight_decay', type=float, default=0.01) # do not modify      
    parser.add_argument('--data_dir', type=str, default="/dataset_path") 
    parser.add_argument(
        '--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        dest='log_level', default='INFO',
        help='logging level for the trainer'
    ) # do not modify
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    # device idx
    opt.device_idx = list(map(int, opt.device.split(',')))
    return parser.parse_args()

    
if __name__ == "__main__":
    
    opt = parse_args()
    
    set_logger("segformer", opt.log_level)
    logging = logging.getLogger("segformer")
    logging.propagate = False
    
    main(opt)