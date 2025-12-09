import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import os
import cv2

from get_args import get_args
from swin_module_bw import *
from dataset import CIFAR10, ImageNet, Kodak
from utils import *

def save_image(tensor, path):
    # tensor: (C, H, W)
    img = as_img_array(tensor)
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0)) # (H, W, C)
    img = img.astype(np.uint8)
    # Convert RGB to BGR for cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def save_comparison(original, reconstructions, bws, path):
    def process_img(tensor, label):
        img = as_img_array(tensor).cpu().numpy()
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Resize for better visibility (32x32 is too small)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        # Add label with black background for readability
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (0, 0), (w + 10, h + 10), (0, 0, 0), -1)
        cv2.putText(img, label, (5, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    img_list = [process_img(original, 'Orig')]
    
    for i, recon in enumerate(reconstructions):
        img_list.append(process_img(recon, f'BW={bws[i]}'))
        
    # Concatenate horizontally
    comparison = np.hstack(img_list)
    cv2.imwrite(path, comparison)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###### Parameter Setting
    args = get_args()
    args.device = device

    job_name = 'JSCC_swin_adapt_lr_' + args.channel_mode +'_dataset_'+str(args.dataset) + '_link_qual_' + str(args.link_qual) + '_n_trans_feat_' + str(args.n_trans_feat)\
                 + '_hidden_size_' + str(args.hidden_size) + '_n_heads_' + str(args.n_heads) + '_n_layers_' + str(args.n_layers) +'_is_adapt_'+ str(args.adapt)


    if args.adapt:
        job_name = job_name + '_link_rng_' + str(args.link_rng)  + '_min_trans_feat_' + str(args.min_trans_feat) + '_max_trans_feat_' + str(args.max_trans_feat) + \
                    '_unit_trans_feat_' + str(args.unit_trans_feat) + '_trg_trans_feat_' + str(args.trg_trans_feat) 

    print(f"Loading model: {job_name}")

    ###### The JSCC Model using Swin Transformer ######
    enc_kwargs = dict(
            args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
            embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[0], args.depth[1]], num_heads=[args.n_heads, args.n_heads],
            window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

    dec_kwargs = dict(
            args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
            embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[1], args.depth[0]], num_heads=[args.n_heads, args.n_heads],
            window_size=args.window_size, mlp_ratio=args.mlp_ratio, norm_layer=nn.LayerNorm, patch_norm=True,)

    source_enc = Swin_Encoder(**enc_kwargs).to(args.device)
    source_dec = Swin_Decoder(**dec_kwargs).to(args.device)

    jscc_model = Swin_JSCC(args, source_enc, source_dec)

    # Load weights
    try:
        _ = load_weights(job_name, jscc_model, device=device)
        print("Weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights not found for {job_name}")
        return

    jscc_model.eval()

    # Load dataset
    # Using EVALUATE set as in run_swin_adapt.py
    eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')
    eval_loader = data.DataLoader(
        dataset=eval_set,
        batch_size=1, # Process one by one for saving
        shuffle=False,
        num_workers=2
    )

    output_dir = 'inference_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving images to {output_dir}...")

    # Number of images to process
    num_images_to_save = 10 
    
    # Bandwidths to test
    bandwidths = range(args.min_trans_feat, args.max_trans_feat + 1)
    print(f"Simulating bandwidths (channels): {list(bandwidths)}")

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(eval_loader)):
            if i >= num_images_to_save:
                break

            images = images.to(args.device).float()
            
            reconstructions = []
            for bw in bandwidths:
                # Run model
                output = jscc_model(images, bw, snr = args.link_qual)
                reconstructions.append(output[0])

            # Save comparison
            save_comparison(images[0], reconstructions, list(bandwidths), 
                          os.path.join(output_dir, f'comparison_{i}.png'))

    print("Done.")

if __name__ == '__main__':
    main()
