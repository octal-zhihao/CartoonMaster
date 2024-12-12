import itertools
import numpy as np
import torch
from PIL import Image
from torch import nn
from training.models.DDPM.nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from training.models.DDPM.utils.utils import postprocess_output, show_config
import os

"""
单独开一个文件改一下传参方式，免得把代码弄乱了
"""


class Diffusion(object):
    _defaults = {
        "model_path": 'training/my_checkpoint/diffusion_model_last_epoch_weights.pth',
        "channel": 128,
        "input_shape": (96, 96),
        "schedule": "linear",
        "num_timesteps": 1000,
        "schedule_low": 1e-4,
        "schedule_high": 0.02,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            if name in ["schedule_low", "schedule_high"]:
                setattr(self, name, float(value))
            else:
                setattr(self, name, value)
            self._defaults[name] = value
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )

        self.net = GaussianDiffusion(UNet(3, self.channel), self.input_shape, 3, betas=betas)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    def generate_1x1_image(self, save_path):
        with torch.no_grad():
            randn_in = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))

            test_images = self.net.sample(1, randn_in.device, use_ema=False)
            test_images = postprocess_output(test_images[0].cpu().data.numpy().transpose(1, 2, 0))

            Image.fromarray(np.uint8(test_images)).save(save_path)


def predict_demo(gen_num=1, **kwargs):
    ddpm = Diffusion(**kwargs)
    save_dir = kwargs.get("save_dir", "results/predict_out")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(gen_num):
        save_path_1x1 = os.path.join(save_dir, f"predict_1x1_result_{i}.png")
        print(f"Generate 1x1 image {i + 1}...")
        ddpm.generate_1x1_image(save_path_1x1)
        print(f"Generate 1x1 image {i + 1} Done. Saved to: {save_path_1x1}")


if __name__ == "__main__":
    predict_demo(save_dir="G:\Hod\CartoonMaster\Web\front_end\static", model_path="G:\Hod\CartoonMaster\training\my_checkpoint\diffusion_model_last_epoch_weights.pth", cuda=True)
