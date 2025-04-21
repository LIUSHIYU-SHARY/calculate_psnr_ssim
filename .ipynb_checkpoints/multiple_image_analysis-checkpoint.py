import os
import cv2
import numpy as np
import _utils_calculate_psnr_ssim as util
from pytorch_fid import fid_score

def average_metrics(folder1, folder2):
    filenames1 = {os.path.basename(f) for f in os.listdir(folder1) if f.endswith('.jpg')}
    filenames2 = {os.path.basename(f) for f in os.listdir(folder2) if f.endswith('.jpg')}
    
    common_filenames = filenames1.intersection(filenames2)
    total_psnr, total_ssim, total_lpips, total_dice, total_pearson = 0, 0, 0, 0, 0
    count = 0
    
    for filename in common_filenames:
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is not None and img2 is not None:
            crop_border = 0
            psnr = util.calculate_psnr(img1, img2, crop_border, test_y_channel=True)
            ssim = util.calculate_ssim(img1, img2, crop_border, test_y_channel=True)
            lpips = util.calculate_lpips(img1, img2)
            dice = util.dice_coefficient(img1, img2)
            pearson = util.pearson_correlation(img1, img2)
            
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            total_dice += dice
            total_pearson += pearson
            count += 1
    
    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count
        avg_dice = total_dice / count
        avg_pearson = total_pearson / count
        
        print(f'Average PSNR: {avg_psnr:.6f} dB')
        print(f'Average SSIM: {avg_ssim:.6f}')
        print(f'Average LPIPS: {avg_lpips:.6f}')
        print(f'Average Dice: {avg_dice:.6f}')
        print(f'Average Pearson Correlation: {avg_pearson:.6f}')
    else:
        print("No matching files found in both folders.")


# Example usage
# folder1 = '../autodl-tmp/easy/generate_image/Ugatit/test'
# folder2 = '../autodl-tmp/easy/generate_image/Ugatit/compare'
folder1 = '../autodl-tmp/temp_data/our_data'
folder2 = '../autodl-tmp/temp_data/test_B'
fid_value=util.calculate_fid(folder2, folder1)
average_metrics(folder1, folder2)
print(f'fid value: {fid_value:.6f}')