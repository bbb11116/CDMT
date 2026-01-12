from __future__ import print_function
import argparse
import os
import time, platform
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import TestDataset
from utils.loss2 import *
from model.model import LDC_side_lifting
from utils.img_processing import (save_image_batch_to_disk)

IS_LINUX = True if platform.system()=="Linux" else False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LDC trainer.')

    # Data parameters  "E:\CS\ys\0_data_YS\imgs\test"
    parser.add_argument('--model_path', type=str,
                        default=r'D:\chens\project\LDC_SED\checkpoints\checkpoints_circle_3\data_test\199\199_model.pth',
                        help='模型文件路径')
    parser.add_argument('--test_dir', type=str,
                        default=r"D:\chens\Data\data_test\imgs\train\real",
                        #required=True,
                        help='测试图像目录')
    parser.add_argument('--height', type=int,
                        default=1200, help='测试图像高度')
    parser.add_argument('--width', type=int,
                        default=1600, help='测试图像宽度')
    parser.add_argument('--output_dir',
                        type=str,
                        default=r"E:\data_SED_result",
                        help='the path to output the results.')
    parser.add_argument('--is_testing',type=bool,
                        default=True,
                        help='Script in testing mode.')
    parser.add_argument('--predict_all',
                        type=bool,
                        default=False,
                        help='True: Generate all LDC outputs in all_edges ')
    parser.add_argument('--circle',
                        type=bool,
                        default=False,
                        help='True: 将圆画在图上')
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='use previous trained data')  # Just for test
    parser.add_argument('--workers',
                        default=0,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--tensorboard',type=bool,
                        default=True,
                        help='Use Tensorboard for logging.'),
    parser.add_argument('--img_width',
                        type=int,
                        default=1600,
                        help='Image width for training.') # BIPED 352 BSDS 352/320 MDBD 480
    parser.add_argument('--img_height',
                        type=int,
                        default=1200,
                        help='Image height for training.') # BIPED 480 BSDS 352/320
    parser.add_argument('--channel_swap',
                        default=[2, 1, 0],
                        type=int)
    parser.add_argument('--resume_chpt',
                        default='result/resume/',
                        type=str,
                        help='resume training')
    parser.add_argument('--crop_img',
                        default=False,
                        type=bool,
                        help='If true crop training images, else resize images to match image width and height.')
    parser.add_argument('--mean_pixel_values',
                        default=[103.939,116.779,123.68,137.86],
                        type=float)  # [103.939,116.779,123.68,137.86] [104.00699, 116.66877, 122.67892]

    args = parser.parse_args()
    return args

def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")
    if not args.test_dir or not os.path.isdir(args.test_dir):
        print(f"错误: 无效的测试目录: {args.test_dir}")
        return
    checkpoint_path = args.model_path

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda:0')

    # Instantiate model and move it to the computing device
    model = LDC_side_lifting().to(device)    # 通过调用 .to(device) 方法，模型的参数和缓存将被移动到指定的设备上，以便在该设备上进行计算
    dataset_test = TestDataset(
        data_root=args.test_dir,
        test_data="CLASSIC",
        mean_bgr=args.mean_pixel_values[0:3] if len(
            args.mean_pixel_values) == 4 else args.mean_pixel_values,
        img_height=args.height,
        img_width=args.width,
        arg=args  # 传递参数对象
    )
    dataloader_test = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}")
    test(checkpoint_path, dataloader_test, model, device, output_dir, args)

def test(checkpoint_path, dataloader, model, device, output_dir, args):
    circle_list = None
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            print(f"{file_names}: {images.shape}")
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()    # 确保前一cuda操作的结果已经准备好，避免存在数据竞争和错误
            preds, circle_list = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     circle_list,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()        # 释放cuda上的缓存空间
    total_duration = np.sum(np.array(total_duration))
    print("FPS: %f.4" % (len(dataloader)/total_duration))
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")
if __name__ == '__main__':
    args = parse_args()
    main(args)