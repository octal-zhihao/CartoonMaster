# CartoonMaster
一款专注于生成高质量卡通图像的“大师”工具

##  tips
Web文件夹用于存放各位的Web开发代码，你们按需改名字就行。

模型训练代码全部放在training/下

xyx如果不熟悉我的项目结构，先自行在training/下新建文件夹来做就行，后续可以由我来整合。

（不过最近备赛有点忙，如果方便的话，尽量用lightning写下

### git使用建议：

https://blog.csdn.net/m0_56676311/article/details/135509261

https://blog.octalzhihao.top/posts/9200.html

> （仅供参考，大家可以按个人习惯和方便程度来）

## Temporary: 项目计划

- 必做方案GAN（生成式对抗网络）：

可参考仓库：[anime_avatar_gen](https://github.com/xiaoyou-bilibili/anime_avatar_gen)

最基础的DC-GAN肯定得实现

加分项：GAN的各种变种/优化（可能会需要基于一些论文来做）
eg. [DF-GAN](https://github.com/tobran/DF-GAN)、[CycleGAN](https://github.com/junyanz/CycleGAN)、[StackGAN](https://github.com/hanzhanggit/StackGAN/tree/master)

[GAN（生成对抗网络）的系统全面介绍（醍醐灌顶）](https://blog.csdn.net/m0_61878383/article/details/122462196)

https://blog.csdn.net/weixin_43334693/article/details/135271954

[GAN生成对抗网络原理推导+代码实现](https://blog.csdn.net/sdksdf/article/details/135068553?app_version=6.3.1&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22135068553%22%2C%22source%22%3A%22sdksdf%22%7D&uLinkId=usr1mkqgl919blen&utm_source=app)

[生成对抗网络GAN原理解析](https://www.bilibili.com/video/BV1nA4m1N74j/?vd_source=cc7c95ecf39d641dd549950fb1aa6069)


深度学习部分项目结构

```bash
training/
├── data/   # 数据集处理相关代码
│   ├── __init__.py
│   ├── data_interface.py # 数据接口类，用于数据的加载与预处理
│   └── cartoon_dataset.py # 构建Dataset
├── dataset/ # 存放数据集
│   ├──── cartoon_color/
│   └──── cartoon_gray/
├── eval_script.py # 评测代码
├── models/   # 模型相关代码
│   ├── __init__.py
│   ├── model_interface.py # 模型接口类
│   └── DC_GAN.py
├── utils/ # 工具函数和辅助脚本
│   ├── __init__.py
│   └── ... # 待补充
├── main.py
├── requirements.txt # 项目环境依赖
└── README.md
```


