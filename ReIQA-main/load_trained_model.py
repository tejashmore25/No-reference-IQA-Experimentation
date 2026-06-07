import torch
from options.train_options import TrainOptions
from networks.build_backbone import build_model

def load_reiqa_model(ckpt_path):
    # parse SAME options as training
    args = TrainOptions().parse()
    
    # # build model
    # model, _ = build_model(args)
    
    # # load checkpoint
    # checkpoint = torch.load(ckpt_path, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    
    # model.eval()
    return args

if __name__ == "__main__":
    ckpt_path = './reiqa_ckpts/content_aware_r50.pth'
    # model = load_reiqa_model(ckpt_path)
    # print("Model loaded successfully.")
    args = load_reiqa_model(ckpt_path)
    print(args)