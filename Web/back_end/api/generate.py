from flask import request, Response
from Web.back_end.api import api
import json
from training.models import MInterface
from training.data import DInterface
from training.main import predict_demo
import yaml


def set_args(model_name, data):
    """根据config文件和传入的data设置参数"""
    args = {}
    with open("Web/config.yaml", 'r') as config_file:
        # 加载配置文件
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        args.update(config['model'][model_name])
    return args


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
            "gen_std": 1
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
    if 'DC_GAN' in data.get("model"):
        print('正在调用DC_GAN生成图片')
        args = set_args(model_name="DCGAN", data=data)
        model = MInterface.load_from_checkpoint(**args)
        model.eval()
        predict_demo(model, args['latent_dim'], save_dir="Web/front_end/static")

    # 返回json类型字符串
    result = {
        "res": ["./static/gan_img.png"]
    }
    return Response(json.dumps(result, ensure_ascii=False), mimetype='application/json')
