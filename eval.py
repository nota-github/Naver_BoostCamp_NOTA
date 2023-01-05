import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from util.data import generate_loader
from util.utils import label_accuracy_score, add_hist
from segformer import SegformerForSemanticSegmentation
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_dir", type=str,  default='/dataset_path')
    parser.add_argument('--pretrain', type=str, default='nvidia/segformer-b2-finetuned-ade-512-512')
    return parser.parse_args()


@torch.no_grad()
def main():
    opt = parse_args()
    
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(opt.pretrain)
    eval_loader = generate_loader(opt, 'val')
    model.eval()
    model = model.to(dev)
    evaluate(eval_loader, model, dev)

def evaluate(loader, model, dev):
    num_labels = model.config.num_labels
    hist = np.zeros((num_labels, num_labels))
    for inputs in tqdm(loader):
        imgs = inputs['pixel_values'].to(dev)
        outputs = model(pixel_values=imgs)
        # Forward pass
        logits = outputs.logits
        # compute miou
        labels = inputs['labels'].to(dev)
        preds = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        hist = add_hist(hist, labels, preds, n_class=num_labels)
            
    acc, _, miou, _, IoU = label_accuracy_score(hist)
    IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , list(model.config.label2id.keys()))]
    print(f'Accuracy : {round(acc, 4)}, mIoU: {round(miou, 4)}')
    print(f'IoU by class : {IoU_by_class}')
        
    
if __name__ == '__main__':
    main()