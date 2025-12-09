import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import os
import cv2
import matplotlib.pyplot as plt

from get_args import get_args
from swin_module_bw import *
from utils import *

def save_comparison(original, reconstructions, bws, path):
    def process_img(tensor, label):
        img = as_img_array(tensor).cpu().numpy()
        img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Add label with black background for readability
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img, (0, 0), (w + 10, h + 10), (0, 0, 0), -1)
        cv2.putText(img, label, (5, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    img_list = [process_img(original, 'Original')]
    
    for i, recon in enumerate(reconstructions):
        img_list.append(process_img(recon, f'BW={bws[i]}'))
        
    # Arrange in 2 rows
    n_total = len(img_list)
    n_cols = (n_total + 1) // 2
    
    row1_list = img_list[:n_cols]
    row2_list = img_list[n_cols:]
    
    # Pad the second row with black image if needed
    while len(row2_list) < n_cols:
        h, w, c = img_list[0].shape
        row2_list.append(np.zeros((h, w, c), dtype=np.uint8))
        
    row1 = np.hstack(row1_list)
    row2 = np.hstack(row2_list)
    comparison = np.vstack([row1, row2])
    cv2.imwrite(path, comparison)

def create_hd_image(height=512, width=768):
    # Create a blank image
    img = np.zeros((height, width, 3), np.uint8)
    img.fill(255) # White background

    # Draw some shapes
    cv2.circle(img, (width//2, height//2), 100, (0, 0, 255), -1)
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(img, (width-200, height-200), (width-50, height-50), (255, 0, 0), -1)
    
    # Draw some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'DeepJSCC HD Test', (width//2 - 200, 100), font, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, 'High Resolution', (width//2 - 150, height - 50), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

    # Add some grid lines
    for i in range(0, width, 50):
        cv2.line(img, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 50):
        cv2.line(img, (0, i), (width, i), (200, 200, 200), 1)

    # Convert to tensor format (C, H, W)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    return torch.from_numpy(img).float().unsqueeze(0) # (1, C, H, W)

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
    # Note: We initialize with default size, but the model should adapt
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

    output_dir = 'inference_results_hd'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving images to {output_dir}...")

    # Create HD image
    # Load specific image
    img_path = 'wallhaven-qr27rq.jpg'
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    print(f"Loading image from {img_path}...")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize/Crop to be multiples of 64
    h, w, _ = img.shape
    new_h = (h // 64) * 64
    new_w = (w // 64) * 64
    img = cv2.resize(img, (new_w, new_h))
    
    # Convert to tensor
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    hd_image = torch.from_numpy(img).float().unsqueeze(0) # (1, C, H, W)
    
    hd_image = hd_image.to(args.device)

    # Bandwidths to test
    bandwidths = range(args.min_trans_feat, args.max_trans_feat + 1)
    print(f"Simulating bandwidths (channels): {list(bandwidths)}")

    psnr_list = []
    with torch.no_grad():
        reconstructions = []
        for bw in bandwidths:
            print(f"Processing BW={bw}...")
            # Run model
            # The model's forward pass calls update_resolution automatically
            output = jscc_model(hd_image, bw, snr = args.link_qual)
            reconstructions.append(output[0])
            
            # Calculate PSNR
            # calc_psnr expects batch, so we pass output and hd_image directly
            p = calc_psnr(output, hd_image)[0]
            # p might be a tensor or float depending on implementation details, ensure it's float
            val = p.item() if hasattr(p, 'item') else p
            psnr_list.append(val)
            print(f"  PSNR: {val:.2f} dB")

        # Save comparison
        save_comparison(hd_image[0], reconstructions, list(bandwidths), 
                      os.path.join(output_dir, 'hd_comparison.png'))
        
        # Plot PSNR
        plt.figure(figsize=(8, 6))
        plt.plot(list(bandwidths), psnr_list, marker='o', linestyle='-', color='b')
        plt.title(f'PSNR vs Bandwidth (SNR={args.link_qual}dB)')
        plt.xlabel('Bandwidth (Feature Channels)')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'psnr_plot.png'))
        plt.close()
        print(f"PSNR plot saved to {os.path.join(output_dir, 'psnr_plot.png')}")

    print("Done.")

if __name__ == '__main__':
    main()
