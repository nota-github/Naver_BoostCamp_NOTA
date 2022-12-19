
import os
import json
import torch
import argparse
from thop import profile
from segformer import SegformerForSemanticSegmentation, SegformerConfig

def main(opt):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(opt.data_dir, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    
    ### from scratch training
    model = SegformerForSemanticSegmentation(
        SegformerConfig(
            num_labels=len(id2label), 
            id2label=id2label, 
            label2id=label2id, 
            ignore_mismatched_sizes=True)
    )
    model = model.to(dev)
    input_vector = torch.randn(1, 3, 512, 512)
    macs, params = profile(model.cpu(), inputs= (input_vector,))
    print('flops',macs)
    print('params',params)
    print('FLOPs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/dataset_path")
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_args()
    main(opt)