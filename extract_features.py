import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from FMF.models import build_model
from FMF.datasets import build_dataset
from FMF.utils.parser import parse_args, load_config


def extract_one_split(model, dataloader, device, desc="Processing"):   
    
    all_features1 = []  # 视频特征
    all_features2 = []  # 电流特征
    all_labels = []     # 标签

    with torch.no_grad():
        for xy in tqdm(dataloader, desc=desc, ncols=80):
            # xy: (video, current, label)
            if len(xy) != 3:
                continue

            x1 = xy[0].to(device)    # (B, C, T, H, W)
            x2 = xy[1].to(device)    # (B, C, T)
            y = xy[2].cpu().numpy()  # (B,)

            try:
                feat1, feat2 = model(x1, x2)  # 提取 [CLS] 特征
                all_features1.append(feat1.cpu().numpy())
                all_features2.append(feat2.cpu().numpy())
                all_labels.append(y)
            except Exception as e:
                print(f"Error during forward: {str(e)}")
                continue

   
    features1 = np.concatenate(all_features1, axis=0) if all_features1 else np.array([])
    features2 = np.concatenate(all_features2, axis=0) if all_features2 else np.array([])
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])

    return features1, features2, labels


def main():
    args = parse_args()
    cfg = load_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './extracted_features'
    os.makedirs(save_dir, exist_ok=True)

    print(f'[INFO] Building model...')
    model = build_model(cfg)

   # 加载预训练权重
    pretrained_path = cfg.MODEL.PRETRAINED 
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Model weights not found: {pretrained_path}")

    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 动态替换 forward 函数 以提取 [CLS] token
    def new_forward(self, x1, x2):
        x1 = self.to_patch_embedding1(x1)  # (B, L1, D)
        x2 = self.to_patch_embedding2(x2)  # (B, L2, D)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1_out, x2_out = self.transformer(x1, x2)
        return x1_out[:, 0], x2_out[:, 0]  # 返回 [CLS] token

    bound_method = new_forward.__get__(model, model.__class__)
    model.forward = bound_method

    # 分别提取 train 和 test 特征
    for split in ['train', 'test']:
        print(f'\n[INFO]  Extracting {split.upper()} features...')
        dataset = build_dataset(name=cfg.TEST.DATASET, cfg=cfg, split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        feats1, feats2, labels = extract_one_split(model, dataloader, device, desc=f"Extract {split}")
        
        np.save(os.path.join(save_dir, f'features_video_{split}.npy'), feats1)
        np.save(os.path.join(save_dir, f'features_current_{split}.npy'), feats2)
        np.save(os.path.join(save_dir, f'labels_{split}.npy'), labels)

        print(f" {split.capitalize()} features saved:")
        print(f"   Video: {feats1.shape}")
        print(f"   Current: {feats2.shape}")
        print(f"   Labels: {labels.shape}")

    print(f'\n All features saved to: {save_dir}/')

if __name__ == '__main__':
    main()