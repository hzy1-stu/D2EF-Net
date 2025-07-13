import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from my_math import get_rmse, get_psnr, get_ssim, complex2real, real2complex, sos, torch_fft2c
import shutil
import argparse
from networks import MICCANlong
from utils import *
import random
import scipy.io as io
from data import DataPrepare, savefig, psnr_plot, dice_plot, acc_plot, loss_plot, ssim_plot, save_result, plot_confusion_matrix, roc
from time import time
from tqdm import tqdm
from metrics import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# 定义函数，保存最新和最佳的模型
# checkpoint用来保存每次训练好的模型参数(也可以理解为保存最新的模型)，而model_best是从保存好的模型参数中找出最佳模型并保存
def save_checkpoint(state, save_path, is_best, filename='checkpoint.pth.tar'):
    # 把序列化的对象保存到硬盘。它利用了 Python 的 pickle 来实现序列化。模型、张量以及字典都可以用该函数进行保存；
    torch.save(state, os.path.join(save_path,filename))

    if is_best:
        # shutil.copyfile(file1,file2)
        # file1为需要复制的源文件的文件路径,file2为目标文件的文件路径+文件名.
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


def validate(val_loader, model, cri, epoch, device, test=False):
    global save_path
    # 运用AverageMeter函数计算每一个epoch的平均损失，即将每个样本的损失取一下平均。
    # 调用此函数的时候，要先声明一下
    val_loss = AverageMeter()
    # model.eval()的作用是 不启用 Batch Normalization 和 Dropout。
    # 如果模型中有 BN 层（Batch Normalization）和 Dropout，在 测试时 添加 model.eval()。
    # model.eval() 是保证 BN 层能够用 全部训练数据 的均值和方差，即测试过程中要保证 BN 层的均值和方差不变。对于 Dropout，model.eval() 是利用到了 所有 网络连接，即不进行随机舍弃神经元。
    model.eval()

    allpsnr = []
    allssim = []
    allacc = []
    allpred = []
    alltrue = []
    i = 0

    bar = 'Test' if test else 'Validate'

    "定义分类损失"
    class_ce = LabelSmoothingCrossEntropy()

    with torch.no_grad():
        for u_data, mask, data, u_kdata, data_label in tqdm(val_loader, desc='%s  |  Epoch-' % bar + str(epoch+1)):
            u_data = u_data.float().to(device)
            mask = mask.float().to(device)
            data = data.float().to(device)
            u_kdata = u_kdata.float().to(device)
            data_label = data_label.float().to(device)

            # model forward
            # 模型进行前向传播
            "图像域重建结果"
            re = model(u_data, u_kdata, mask, True)
            "K空间域重建结果"
            k_re = complex2real(torch_fft2c(real2complex(re)))
            "将重建结果放入分类网络的到分类结果"
            class_re = model(re, k_re, mask, False)

            outputs_softmax = torch.softmax(class_re, dim=1)

            "将重建结果转为复数并进行sos运算，以便后续计算指标和保存数据"
            re = sos(real2complex(re))

            # 计算损失值
            "重建损失"
            loss_rec = cri(re, data)
            "分类损失"
            loss_class = class_ce(class_re, data_label.long())
            "总损失"
            loss = 0.7 * loss_rec + 0.3 * loss_class

            "计算分类acc指标"
            acc = accuracy_score(data_label.cpu().numpy(), torch.argmax(torch.softmax(class_re, dim=1), dim=1).cpu().numpy())

            "保存数据"
            rec = torch.squeeze(re)
            raw = torch.squeeze(data)
            under = torch.squeeze(sos(real2complex(u_data)))
            label = data_label

            # 计算相应的评估值
            psnr = get_psnr(rec, raw)
            ssim = get_ssim(rec, raw)
            allpsnr.append(psnr)
            allssim.append(ssim.item())
            allacc.append(acc)
            allpred.extend(outputs_softmax.cpu().numpy())  # 存储概率而非类别
            alltrue.extend(label.cpu().numpy())

            # 保存测试的结果
            # save result when testing
            if test:
                save_result(raw, under, rec, label, save_path, i)

            # 记录验证的损失
            # AverageMeter是根据batch_size的大小来计算损失，所以括号里面会添加u_data.size(0)
            # record validation loss
            val_loss.update(loss.item(), u_data.size(0))

            i = i+1
        # 记录测试过程中每一批次的评分到txt文件中
        # record score of every batch in txt while testing
        # array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会
        if test:
            # np.savetxt(保存路径，保存数据，使用默认分割符（空格）并保留四位小数)
            np.savetxt(os.path.join(save_path, 'PSNR_batches_test.txt'), np.asarray(allpsnr), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'SSIM_batches_test.txt'), np.asarray(allssim), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'acc_batches_test.txt'), np.asarray(allacc), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'pred_batches_test.txt'), np.asarray(allpred), fmt='%.4f')
            np.savetxt(os.path.join(save_path, 'true_batches_test.txt'), np.asarray(alltrue), fmt='%.4f')

            plot_confusion_matrix(save_path, alltrue, [np.argmax(p) for p in allpred])
            roc(alltrue, allpred)

        # print out average scores
        # 输出质量评估分数的平均值
        avgpsnr = np.mean(np.asarray(allpsnr))
        avgssim = np.mean(np.asarray(allssim))
        avgacc = np.mean(np.asarray(allacc))

        print(' * Epoch-'+str(epoch+1)+'\tAverage Validation Loss {:.9f}'.format(val_loss.avg))
        print(' * Epoch-'+str(epoch+1)+'\tAverage PSNR {:.4f}'.format(avgpsnr))
        print(' * Epoch-'+str(epoch+1)+'\tAverage SSIM {:.4f}'.format(avgssim))
        print(' * Epoch-' + str(epoch + 1) + '\tAverage acc {:.4f}'.format(avgacc))
        print(classification_report(alltrue, [np.argmax(p) for p in allpred], target_names=['Meningioma', 'Glioma', 'Pituitary'], digits=5))

        return val_loss.avg, avgpsnr, avgssim, avgacc


# data数量为10，每个batch_size大小为1，所以总共有10个batch。epoch默认值设置为100，所以相当于内循环里面把每一个batch拿出来训练，而外循环里面把内循环循环100次
parser = argparse.ArgumentParser(description='Main function arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', default=False, help='train or test')
parser.add_argument('--multi', default=False, help='train on multi gpu or not')
parser.add_argument('--seed', default=6, type=int, help='random seed')
parser.add_argument('--loss', default='l1', type=str, help='loss function')
parser.add_argument('--nblock', default=1, type=int, help='number of block')
parser.add_argument('--gpuid', default='0', type=str, help='gpu id')  # 这里的默认值0指的是GPU的编号
parser.add_argument('--bs', default=6, type=int, help='batchsize')
parser.add_argument('--epoch', default=50, type=int, help='number of epoch')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--savepath', default='result', type=str, help='save file path')
parser.add_argument('--maskname', default='mask_1DRU_R3_512x512_ACS22', type=str, help='mask name')


# global variables
"""全局变量"""
n_iter = 0
best_loss = -1

"""主函数（Pytroch利用GPU训练模型需要将设计好的模型和数据放入指定的GPU上，至于损失函数个人认为不需要放到GPU上，当然放上去也不会报错）"""
# main function
def main():
    # H获取当前时间戳（从世界标准时间的1970年1月1日00：00：00开始到当前这一时刻为止的总秒数），即计算机内部时间值，浮点数。
    time0 = time()
    """运用argparse模块，是所有参数增删改全在同一区域中进行"""
    args = parser.parse_args()
    # 设置当前使用的GPU设备仅为0（gpuid = 0）号设备  设备名称为'/gpu:0'
    # CUDA_VISIBLE_DEVICES 表示当前可以被python环境程序检测到的显卡Z
    # 进行指定使用设备，这样会修改pytorch感受的设备编号，pytorch感知的编号还是从device:0开始。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    # Y不管外部变量的类型是什么，如果在函数内部想对它做赋值操作就必须使用global声明。
    global save_path, n_iter, best_loss

    # set random seed
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
    torch.cuda.manual_seed(args.seed)

    # save model in save_path 模型保存到定义好的save_path路径中
    save_path = args.savepath
    # 通过str.format方法通过字符串中的花括号 {} 来识别替换字段，从而完成字符串的格式化。
    print('The result will be saved in {}'.format(save_path))  # 程序运行后，第一行显示的是该print
    # os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。如果存在，返回True，反之False
    if not os.path.exists(save_path):
        # os.makedirs用来创建多层目录，save_path为目录名
        os.makedirs(save_path)

    # specify network structure
    network = MICCANlong(2, 2, args.nblock)

    # whether is using multiple gpu
    # 判断是否可以在多块GPU上并行计算
    if args.multi is not False:
        network = nn.DataParallel(network)

    # specify loss function
    if args.loss == 'l2':
        loss = nn.MSELoss()
    if args.loss == 'l1':
        loss = nn.L1Loss()

    # initialize optimizer and schedule loss decay
    optimizer = Adam(network.parameters(), lr=args.lr)
    # 根据epoch次数来调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # initialize device
    # 初始化设备
    print('CUDA is available:' + str(torch.cuda.is_available()) + '\tCUDA_version:' + str(torch.version.cuda))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)

     testloder = DataLoader(DataPrepare('test', args.maskname), batch_size=1, shuffle=False)
     # 在Pytorch中构建好一个模型后，一般需要进行预训练权重中加载。torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
     # ['state_dict']意思是加载保存在state_dict里面的参数
     network.load_state_dict(torch.load(args.savepath + '/model_best.pth.tar')['state_dict'])
     validate(testloder, network, loss, 0, device, test=True)
     print('Finish   |   Consume:{:.2f}min'.format((time() - time0) / 60))



if __name__ == '__main__':
    main()