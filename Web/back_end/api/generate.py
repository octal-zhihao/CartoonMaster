import os
import shutil
from flask import request, Response
from Web.back_end.api import api
import json
from training.models import MInterface
from training.data import DInterface
from training.main import predict_demo as DCGAN_generator
from training.DDPM.predict import predict_demo as DDPM_generator
from training.WGAN.generate import main as WGAN_generator

import yaml


def set_args(model_name, data):
    """根据config文件和传入的data设置参数"""
    args = {}
    with open("Web/config.yaml", 'r') as config_file:
        # 加载配置文件
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        args.update(config['model'][model_name])
    return args


def clear_dir(directory):
    """清空文件夹"""
    # 检查文件夹是否存在
    if os.path.exists(directory) and os.path.isdir(directory):
        # 清除文件夹内的所有文件和子文件夹
        shutil.rmtree(directory)
        # 重新创建文件夹
        os.makedirs(directory, exist_ok=True)

        print(f"已清空 {directory} 文件夹")
    else:
        print(f"{directory} 文件夹不存在")


@api.route('generate', methods=['POST'])
def generate():
    """使用指定模型生成图像，接收表单数据，返回图片路径

        ### args
        |  args          | required | request type | type |  remarks                  |
        |----------------|----------|--------------|------|---------------------------|
        | model          |  true    |    form      | str  | 使用的模型  |
        | gen_search_num |  false   |    form      | int  | 总生成数  |
        | gen_num        |  false   |    form      | int  | 最终图片数 |
        | gen_mean       |  false   |    form      | int  | 噪声均值（暂无）   |
        | gen_std        |  false   |    form      | int  | 噪声方差（暂无）   |

        ### request
        ```
        {
            "model": "DC_GAN",
            "gen_search_num": 10,
            "gen_num": 5,
            "gen_mean": 0,
            "gen_`std": 1
        }
        ```

        ### return
        ```json
        {
            "res": ["./static/gan_img.png"]
        }
        ```
        """
    data = request.form
    save_dir = "Web/front_end/static"
    clear_dir(save_dir)
    if 'DC_GAN' in data.get("model"):
        print('正在调用DC_GAN生成图片')
        args = set_args(model_name="DCGAN", data=data)
        model = MInterface.load_from_checkpoint(**args)
        model.eval()
        DCGAN_generator(model, args['latent_dim'], save_dir="Web/front_end/static")
    if 'DDPM' in data.get("model"):
        print('正在调用DDPM生成图片')
        DDPM_generator(save_dir="Web/front_end/static")
    if 'WGAN' in data.get("model"):
        print('正在调用WGAN生成图片')
        args = set_args(model_name="WGAN", data=data)
        WGAN_generator(cfg=args)
    res = []
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            # 构建相对路径
            relative_path = os.path.relpath(os.path.join(root, file), save_dir)
            # 构建最终路径格式
            final_path = f"./static/{relative_path}"
            res.append(final_path)
    # 返回json类型字符串
    result = {
        "res": res
    }
    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')
