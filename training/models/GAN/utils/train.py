# coding:utf8
import torch as t
import torchvision as tv
import tqdm
import yaml

from training.models.GAN.model.BaseNet import NetD, NetG
from torchnet.meter import AverageValueMeter
import visdom


class Config(object):
    def __init__(self, cfg):
        self.data_path = "../data"  # 数据集存放路径
        self.num_workers = cfg['num_workers']  # 多进程加载数据所用的进程数
        self.image_size = cfg['image_size']  # 图片尺寸
        self.batch_size = cfg['batch_size']  # 一次取多少张图片
        self.max_epoch = cfg['max_epoch']  # 迭代轮次
        self.lr1 = cfg['lr1']  # 生成器的学习率
        self.lr2 = cfg['lr2']  # 判别器的学习率
        self.beta1 = cfg['beta1']  # Adam优化器的beta1参数
        self.gpu = cfg['gpu']  # 是否使用GPU
        self.vis = False  # 是否使用visdom可视化
        self.plot_every = 20  # 每间隔20 batch，visdom画图一次
        self.d_every = cfg['d_every']  # 每1个batch训练一次判别器
        self.g_every = cfg['g_every']  # 每5个batch训练一次生成器
        self.save_every = 10  # 每10个epoch保存一次模型
        self.nz = cfg['nz']  # 噪声维度
        self.ngf = cfg['ngf']  # 生成器feature map数
        self.ndf = cfg['ndf']  # 判别器feature map数
        # 预训练模型，通过预训练模型可以缩短训练周期
        self.netd_path = cfg['discriminator_path']
        self.netg_path = cfg['generator_path']
        self.base = 1 + 2000  # 基础值（根据需要调整）


if __name__ == '__main__':
    with open('../../../config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['model']['GAN']
    opt = Config(cfg)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    if opt.vis:
        vis = visdom.Visdom()  # 默认使用的env是main

    # 构建一个图形变换库，这里是对我们的图像进行处理
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载我们的数据集，ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
    # 这里传入了我们前面的图片路径和变换库，经过这一步操作，我们获取到了数据集
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    # 调用troch的数据加载器，加载我们的数据集
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )
    # 加载我们的网络G表示生成器，D表示判别器
    # 分别传入一个feature map和噪声维度，判别器不需要传
    netg, netd = NetG(opt.ngf, opt.nz), NetD(opt.ndf)

    # 在预训练模型上进行迭代，默认我们不进行迭代，如果需要可以去掉下面的注释
    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    # 把我们的网络移动到CUDA设备上去
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    # 这里我们构造了两个优化器
    # 第一个参数是我们模型的所有参数，然后第二个是学习率，第三个betas表示用于计算梯度的平均和平方的系数，默认: (0.9, 0.999)
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    # 这里我们定义了一个loss函数
    criterion = t.nn.BCELoss().to(device)

    # 这里我们定义了真图片和假图片的标签，t.ones表示生成一个全为1的张量
    # 真图片label为1，假图片label为0
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)

    # 生成随机数张量，这里我们的噪声维度为100维,然后每一维都是1*1大小的矩阵
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # 初始化我们统计量的均值
    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    # 下面开始进行迭代
    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        # 这里加载我们的数据集,tqdm.tqdm在迭代时会显示一个进度条
        print("Epoch {}/{}".format(epoch + 1, opt.max_epoch))
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            # img表示我们的图片,ii表示当前为第几次batch
            real_img = img.to(device)

            # 这里训练我们的判别器
            if ii % opt.d_every == 0:
                # 首先进行梯度下降，把模型的梯度参数初始化为0
                optimizer_d.zero_grad()
                # 首先我们把真的图片的参数传递进去，然后获取损失值
                # 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                # 进行一下权重更新
                error_d_real.backward()

                # 尽可能把假图片判别为错误
                # 这里生成随机噪声
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                # 根据我们前面的噪声，我们生成一张假图片
                fake_img = netg(noises).detach()
                # 这里我们使用判别器来判断我们生成的假图片
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                # 进行单步优化，前面我们损失函数计算好之后就可以调用这个函数了
                optimizer_d.step()

                # 这里我们把损失值放进去
                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            # 这里训练我们的生成器
            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            fix_fake_imgs = netg(fix_noises)
            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                # 可视化
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                print('errord', errord_meter.value()[0])
                print('errorg', errorg_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            print("Saving model...")
            # 保存模型和图片
            # 我们调用torch version的自带的功能去保存图片
            normalized_imgs = (fix_fake_imgs.data[:64] + 1) / 2  # 假设原图像在 [-1, 1] 范围
            tv.utils.save_image(normalized_imgs, 'model/result/%s.png' % (epoch + opt.base), normalize=True)
            # tv.utils.save_image(fix_fake_imgs.data[:64], 'model/gan/%s.png' % epoch, normalize=True,
            #                     range=(-1, 1))
            # 保存我们生成器和判别器的模型
            t.save(netd.state_dict(), 'model/result/netd_%s.pth' % (epoch + opt.base))
            t.save(netg.state_dict(), 'model/result/netg_%s.pth' % (epoch + opt.base))
            # 迭代完后我们重设一下
            errord_meter.reset()
            errorg_meter.reset()
