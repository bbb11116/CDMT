from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



import argparse
import cv2
import numpy as np
import time, platform
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
#from model.mydateset import YSDataset, YSTestDataset
#from utils.dataset import *
from utils.dataset_all import *
from utils.circle_dataset import *
from utils.loss2 import *
from model.model import LDC_side_lifting
from utils.img_processing import (save_image_batch_to_disk,
                                  visualize_result, count_parameters)
from utils.yolo_circleLoss import YoloCircleLoss
IS_LINUX = True if platform.system()=="Linux" else False
def train_one_epoch(epoch, dataloader, model, criterions, optimizer, device,
                    log_interval_vis, tb_writer, args=None):

    imgs_res_folder = os.path.join(args.output_dir, 'current_res')
    os.makedirs(imgs_res_folder,exist_ok=True)

    if isinstance(criterions, list):    # 如果criterions是list,则执行判断
        criterion1, criterion2, criterion3, criterion4, criterion5 = criterions
    else:
        criterion1 = criterions

    # Put model in training mode
    model.train()
    l_weight = [[0.05, 2.], [0.05, 2.],
                [0.01, 1.], [0.01, 1.], [0.01, 4]]
    l_weight0 = [1.0, 1.0, 1.0, 1.3, 1.7]
    loss_avg =[]
    for batch_id, sample_batched in enumerate(dataloader):
        images = sample_batched['images'].to(device)  # BxCxHxW   图像信息
        labels = sample_batched['labels'].to(device)  # BxHxW  labels信息

        preds_list,circle_list = model(images)
        assert len(preds_list) == len(l_weight), "长度不匹配"

        loss_4 = sum([criterion4(preds, labels, l_w) for preds, l_w in zip(preds_list, l_weight0)])  # bdcn_loss2 [1,2,3] TEED
        loss_1 = criterion1(preds_list[-1], labels, l_weight[-1], device)  # cats_loss [dfuse] TEED
        loss_5 = criterion5(preds = circle_list, batch = sample_batched)[0] # yolo_circleLoss [dfuse] TEED
        loss_5 = loss_5.mean()*0.0000001  # 或 .sum()



        # loss = sum([criterion2(preds, labels,l_w) for preds, l_w in zip(preds_list[:-1],l_weight0)]) # bdcn_loss2
        #loss_1 = sum([criterion1(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss   计算损失
        #loss_3 = sum([criterion2(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight0)])
        loss_2 = criterion2(preds_list[-1], labels, l_weight0[-1], device)
        #loss_4 = sum([criterion3(preds, labels, lweight = l_w) for preds, l_w in zip(preds_list, l_weight0)])
        loss = loss_1 + loss_4 * 2 + loss_2*0.25 + loss_5


        optimizer.zero_grad()   # 将梯度归零
        loss.backward()         # 反向传播计算每个参数的梯度值
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()        # 通过梯度下降来执行参数更新
        loss_avg.append(loss.item())
        if tb_writer is not None:
            step_idx = len(dataloader)*epoch + batch_id
            tmp_loss = np.array(loss_avg).mean()            # 计算loss_avg数组的平均损失
            tb_writer.add_scalar('step_loss', tmp_loss, step_idx)   # 将temp_loss添加到tb_writer中，并命名为loss，将其于epoch联系起来

        if batch_id % log_interval_vis == 0:
            res_data = []           # 创建一个res_data数组

            img = images.cpu().numpy()  # 将image转移到cpu上
            res_data.append(img[0])     #将img[2]添加到res_data中

            ed_gt = labels.cpu().numpy()    # 将gt转移到cpu上
            res_data.append(ed_gt[0])       # 将gt[2]添加到res_data中

            # tmp_pred = tmp_preds[2,...]
            for i in range(len(preds_list)):
                tmp = preds_list[i]
                tmp = tmp[0]        # 获得每个tmp的索引为2的元素信息
                # print(tmp.shape)
                tmp = torch.sigmoid(tmp).unsqueeze(dim=0)       # torch.sigmoid(tmp)：应用 sigmoid 函数对 tmp 进行激活操作，将其值限制在0到1之间。
                                                                # tmp.unsqueeze(dim=0)：在 tmp 的第0维度上添加一个维度，将其转换为形状为 (1, ...)  的张量。
                tmp = tmp.cpu().detach().numpy()        # 将张量 tmp 移到 CPU 上，并将其转换为 NumPy 数组。
                res_data.append(tmp)

            vis_imgs = visualize_result(res_data, arg=args)   # res_data中的所有信息展示在一张画布上
            del tmp, res_data

            vis_imgs = cv2.resize(vis_imgs,
                                  (int(vis_imgs.shape[1]*0.8), int(vis_imgs.shape[0]*0.8)))     # 调整图像为原来的80%
            img_test = 'Epoch: {0} Sample {1}/{2} Loss: {3}' \
                .format(epoch, batch_id, len(dataloader), loss.item())

            BLACK = (0, 0, 255)   # 定义文本的颜色
            font = cv2.FONT_HERSHEY_SIMPLEX     # 指定字体
            font_size = 1.1         # 指定文本字体大小
            font_color = BLACK      # 文本颜色
            font_thickness = 2      # 字体粗细程度
            x, y = 30, 30           # 坐标位置
            # 函数将文本img_test绘制在可视化图像vis_imgs上，位于指定的位置(x, y)，使用指定的字体、字体大小、字体颜色和字体粗细
            vis_imgs = cv2.putText(vis_imgs,
                                   img_test,
                                   (x, y),
                                   font, font_size, font_color, font_thickness, cv2.LINE_AA)
            cv2.imwrite(os.path.join(imgs_res_folder, 'results.png'), vis_imgs)
    loss_avg = np.array(loss_avg).mean()
    return loss_avg

def validate_one_epoch(criterions, dataloader, model, device, output_dir, arg=None):
    # XXX This is not really validation, but testing
    if isinstance(criterions, list):    # 如果criterions是list,则执行判断
        criterion1, criterion2, criterion3, criterion4, criterion5 = criterions
    else:
        criterion1 = criterions
    l_weight = [[0.05, 2.], [0.05, 2.],
                [0.1, 1.], [0, 1.], [0.01, 4]]
    l_weight0 = [1.0, 1.0, 1.0, 1.3, 1.7]

    # Put model in eval mode
    model.eval()
    val_loss_avg =[]

    with torch.no_grad():
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            labels = sample_batched['labels'].to(device)
            file_names = sample_batched['im_file']
            image_shape = sample_batched['shape']
            preds_list,circle_list = model(images)
            loss_4 = sum([criterion4(preds, labels, l_w) for preds, l_w in zip(preds_list, l_weight0)])  # bdcn_loss2 [1,2,3] TEED
            loss_1 = criterion1(preds_list[-1], labels, l_weight[-1], device)  # cats_loss [dfuse] TEED
            loss_5 = criterion5(preds = circle_list, batch = sample_batched)[0]
            loss_5 = loss_5.mean()*0.0000001

            # loss = sum([criterion2(preds, labels,l_w) for preds, l_w in zip(preds_list[:-1],l_weight0)]) # bdcn_loss2
            # loss_1 = sum([criterion1(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight)])  # cats_loss   计算损失
            # loss_3 = sum([criterion2(preds, labels, l_w, device) for preds, l_w in zip(preds_list, l_weight0)])
            loss_2 = criterion2(preds_list[-1], labels, l_weight0[-1], device)
            # loss_4 = sum([criterion3(preds, labels, lweight = l_w) for preds, l_w in zip(preds_list, l_weight0)])
            loss =loss_1  +   loss_4  + loss_2*0.25 + loss_5
            val_loss_avg.append(loss.item())
            # print('pred shape', preds[0].shape)
            # 将预测的结果图像存储到对应的文件中
            save_image_batch_to_disk(preds_list[-1],
                                     circle_list,
                                     output_dir,
                                     file_names, img_shape=image_shape,
                                     arg=arg)
    return np.array(val_loss_avg).mean()  # 返回epoch平均损失


def test(checkpoint_path, dataloader, model, device, output_dir, args):
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
            if not args.train_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            print(f"{file_names}: {images.shape}")
            # if batch_id==0:
            #     mac,param = profile(model,inputs=(images,))
            #     end = time.perf_counter()
            #     if device.type == 'cuda':
            #         torch.cuda.synchronize()
            #     preds = model(images)
            #     if device.type == 'cuda':
            #         torch.cuda.synchronize()
            # else:
            end = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize()    # 确保前一cuda操作的结果已经准备好，避免存在数据竞争和错误
            preds = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            tmp_duration = time.perf_counter() - end
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()        # 释放cuda上的缓存空间
    total_duration = np.sum(np.array(total_duration))
    print("******** Testing finished in", args.train_data, "dataset. *****")
    print("FPS: %f.4" % (len(dataloader)/total_duration))
    # print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def testPich(checkpoint_path, dataloader, model, device, output_dir, args):
    # a test model plus the interganged channels
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
            if not args.train_data == "CLASSIC":
                labels = sample_batched['labels'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            images2 = images[:, [1, 0, 2], :, :]  #GBR images2 是一个重新排列通道维度顺序后的张量，形状与 images 相同，但通道维度的顺序变为 [1, 0, 2]。
            # images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images)
            preds2 = model(images2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds, preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished in", args.train_data, "dataset. *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LDC trainer.')
    # Data parameters

    parser.add_argument('--input_dir',        #训练数据路径
                        type=str,
                        default=r'D:\chens\Data\data_test',
                        help='the path to the directory with the input data.')
    parser.add_argument('--json_train',  # 训练数据路径
                        type=str,
                        default=r'train.json',
                        help='the path to the json file.')
    parser.add_argument('--json_test',  # 测试数据路径
                        type=str,
                        default=r'test.json',
                        help='the path to the json file.')
    parser.add_argument('--output_dir',    #训练结果路径
                        type=str,
                        default='checkpoints/checkpoints_circle_5',
                        help='the path to output the results.')
    parser.add_argument('--train_data',
                        type=str,
                        default='data_test',
                        help='Name of the dataset.')
    parser.add_argument('--test_list',
                        type=str,
                        default='image_test.lst',
                        help='Dataset sample indices list.')
    parser.add_argument('--train_list',
                        type=str,
                        default='image_train.lst',
                        help='Dataset sample indices list.')
    parser.add_argument('--is_testing',type=bool,
                        default=None,
                        help='Script in testing mode.')
    parser.add_argument('--predict_all',
                        type=bool,
                        default=False,
                        help='True: Generate all LDC outputs in all_edges ')
    parser.add_argument('--double_img',
                        type=bool,
                        default=False,
                        help='True: use same 2 imgs changing channels')  # Just for test
    parser.add_argument('--resume',
                        type=bool,
                        default=False,
                        help='use previous trained data')  # 是否使用之前的训练数据
    parser.add_argument('--checkpoint_data',
                        type=str,
                        default=r'"F:\checkpoints\all_9_30.pth"',# 权重路径
                        # 权重路径
                        help='Checkpoint path.')
    parser.add_argument('--checkpoint_data1',
                        type=str,
                        default=r'D:\yingsu\train\CS_LDC_1\demo\checkpoints\checkpoints_ALL_9.18\ALL\43\43_model.pth',
                        help='Checkpoint path.')
    parser.add_argument('--test_img_width',
                        type=int,
                        default='1600',
                        help='Image width for testing.')
    parser.add_argument('--test_img_height',
                        type=int,
                        default='1200',
                        help='Image height for testing.')
    parser.add_argument('--res_dir',
                        type=str,
                        default='result',
                        help='Result directory')
    parser.add_argument('--log_interval_vis',
                        type=int,
                        default=1000,
                        help='The NO B to wait before printing test predictions. 200')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        metavar='N',
                        help='Number of training epochs (default: 25).')  # 训练总轮次
    parser.add_argument('--lr', default=8e-4, type=float,  # 5e-5
                        help='Initial learning rate. =5e-5')  # 初始学习率
    parser.add_argument('--lrs', default=[4e-4, 2e-4, 5e-5], type=float,
                        help='LR for set epochs')
    parser.add_argument('--wd', type=float, default=0., metavar='WD',
                        help='weight decay (Good 5e-6)')
    parser.add_argument('--adjust_lr', default=[20, 30, 40], type=int,
                        help='Learning rate step size.')  # [6,9,19] # 调整学习率的epoch节点
    parser.add_argument('--version_notes',
                        default='LDC-BIPED: B4 Exp 67L3 xavier init normal+ init normal CatsLoss2 Cofusion',
                        type=str,
                        help='version notes')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        metavar='B',
                        help='the mini-batch size (default: 8)')
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
                        help='Image width for training.')
    parser.add_argument('--img_height',
                        type=int,
                        default=1200,
                        help='Image height for training.')
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
    # BRIND mean = [104.007, 116.669, 122.679, 137.86]
    # BIPED mean_bgr processed [160.913,160.275,162.239,137.86]
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # Tensorboard summary writer

    tb_writer = None
    training_dir = os.path.join(args.output_dir,args.train_data)
    os.makedirs(training_dir,exist_ok=True)
    if args.is_testing:
        checkpoint_path = os.path.join(args.output_dir, args.train_data,args.checkpoint_data1)

    # 在训练过程中，如果启用了 TensorBoard 日志记录且不处于测试模式，
    # 创建一个 SummaryWriter 对象用于记录训练过程中的指标和数据，并将训练设置信息保存到一个文本文件中
    if args.tensorboard and not args.is_testing:
        # from tensorboardX import SummaryWriter  # previous torch version
        from torch.utils.tensorboard import SummaryWriter # for torch 1.4 or greather
        tb_writer = SummaryWriter(log_dir=training_dir)
        # saving training settings
        training_notes =['LDC,LR= ' + str(args.lr) +
                           ' image size = ' + str(args.img_width)
                          + ' adjust LR=' + str(args.adjust_lr) +' LRs= '
                          + str(args.lrs)]
        info_txt = open(os.path.join(training_dir, 'training_settings.txt'), 'w')
        info_txt.write(str(training_notes))
        info_txt.close()

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda:0')

    # Instantiate model and move it to the computing device
    model = LDC_side_lifting().to(device)    # 通过调用 .to(device) 方法，模型的参数和缓存将被移动到指定的设备上，以便在该设备上进行计算
    model = model.to(device)
    # model = nn.DataParallel(model)
    ini_epoch =0
    if not args.is_testing:     # 训练过程的数据loader
        if args.resume:
            print("Resuming training")
            # checkpoint_path2 = os.path.join(args.output_dir, 'BIPED-54-B4',args.checkpoint_data)
            ini_epoch = 0
            model.load_state_dict(torch.load(args.checkpoint_path,
                                         map_location=device))
        dataset_train = BipedDataset(args.input_dir,
                                     os.path.join(args.input_dir,args.json_train),
                                     img_width=args.img_width,
                                     img_height=args.img_height,
                                     mean_bgr=args.mean_pixel_values[0:3] if len(
                                         args.mean_pixel_values) == 4 else args.mean_pixel_values,
                                     train_mode='train',
                                     arg=args
                                     )

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers,
                                      collate_fn=custom_collate_fn)
    dataset_val = TestDataset(args.input_dir,
                              os.path.join(args.input_dir,args.json_test),
                              test_data=args.train_data,
                              img_width=args.test_img_width,
                              img_height=args.test_img_height,
                              mean_bgr=args.mean_pixel_values[0:3] if len(
                                  args.mean_pixel_values) == 4 else args.mean_pixel_values,
                              test_list=args.test_list, arg=args
                              )

    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.workers,
                                collate_fn=custom_collate_fn)
    # Testing
    if args.is_testing:

        output_dir = os.path.join(args.res_dir, args.train_data+"1"+ args.train_data)
        print(f"output_dir: {output_dir}")
        if args.double_img:
            # run twice the same image changing the image's channels
            testPich(checkpoint_path, dataloader_val, model, device, output_dir, args)
        else:
            test(checkpoint_path, dataloader_val, model, device, output_dir, args)

        # Count parameters:
        num_param = count_parameters(model)
        print('-------------------------------------------------------')
        print('LDC parameters:')
        print(num_param)
        print('-------------------------------------------------------')
        return

    criterion1 = cats_loss #bdcn_loss2
    criterion2 = Dice_loss#cats_loss#f1_accuracy2
    criterion3 = HybridLoss(
        max_epochs=args.epochs,
        scheduler_type='cosine',
        hard_threshold=(0.3, 0.7),
        hard_weight=3.0,
        hard_gamma=2.0
    )
    criterion5 = YoloCircleLoss()
    criterion4 = bdcn_loss2
    criterion = [criterion1, criterion2, criterion3, criterion4,criterion5]
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.wd)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=1e-6,  # ← 关键！别用 1e-3
    #       # ← 强烈建议开启
    #     weight_decay=args.wd  # ← L2 正则，防止过拟合
    # )

    # Count parameters:
    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('LDC parameters:')
    print(num_param)
    print('-------------------------------------------------------')

    # Main training loop
    seed = 1021
    adjust_lr = args.adjust_lr
    k=0
    set_lr = args.lrs#[25e-4, 5e-6]

    loss_history = dict((k, []) for k in ["epoch", "train_loss"])
    val_loss_history= dict((k, []) for k in ["epoch", "val_loss"])

    epochs_bar = tqdm(total=args.epochs - ini_epoch, desc="Overall Progress", unit="epoch", initial=ini_epoch)

    for epoch in range(ini_epoch,args.epochs):
        criterion3.current_epoch = epoch
        if epoch%7 == 0:

            seed = seed+1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # print("------ Random seed applied-------------")
        # adjust learning rate
        if adjust_lr is not None:
            if epoch in adjust_lr:
                lr2 = set_lr[k]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr2
                k+=1
        # Create output directories

        output_dir_epoch = os.path.join(args.output_dir,args.train_data, str(epoch))
        img_test_dir = os.path.join(output_dir_epoch, args.train_data + '_res')
        os.makedirs(output_dir_epoch, exist_ok=True)
        os.makedirs(img_test_dir, exist_ok=True)
        # Training loop
        avg_loss = train_one_epoch(epoch,dataloader_train,
                        model, criterion,
                        optimizer,
                        device,
                        args.log_interval_vis,
                        tb_writer=tb_writer,
                        args=args)

        loss_history["epoch"].append(epoch)
        loss_history["train_loss"].append(avg_loss)

        val_loss = validate_one_epoch(criterion,
                           dataloader_val,
                           model,
                           device,
                           img_test_dir,
                           arg=args)
        val_loss_history["epoch"].append(epoch)
        val_loss_history["val_loss"].append(val_loss)

        # Save model after end of every epoch
        torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                   os.path.join(output_dir_epoch, '{0}_model.pth'.format(epoch)))
        if tb_writer is not None:
            tb_writer.add_scalar('epoch_loss',
                                 avg_loss,
                                 epoch+1)

        #显示进度条
        epochs_bar.set_postfix(loss=f"{val_loss:.4f}", last_learning_rate=f"{optimizer.param_groups[0]['lr']:.8f}")
        #更新进度条（步进1）
        epochs_bar.update(1)
        # print('Last learning rate> ', optimizer.param_groups[0]['lr'])
        if np.isnan(val_loss):
            print("训练已完成")
            break

    import matplotlib
    matplotlib.use('Agg')  # 防止在服务器上出错
    import matplotlib.pyplot as plt
    from datetime import datetime

    # 绘制 loss 曲线
    plt.figure()
    plt.title('Loss during training')
    plt.plot(loss_history["epoch"], loss_history["train_loss"], label="train_loss")
    plt.plot(val_loss_history["epoch"], val_loss_history["val_loss"], label="val_loss", color='darkorange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 在图上添加时间水印（例如右下角）
    plt.text(0.99, 0.01, f'Time: {current_time}',
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

    # 先保存，再显示（如果需要）
    save_path = os.path.join(args.output_dir, 'loss_figure.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高质量保存
    print(f"Loss curve saved to {save_path}")

    plt.show()  # 本地调试时可看到图
    plt.close()  # 可选：释放内存


    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('LDC parameters:')
    print(num_param)
    print('-------------------------------------------------------')

if __name__ == '__main__':
    args = parse_args()
    main(args)
