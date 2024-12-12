import os
import torch
import torchvision
from training.models.WGANGP.net import Discriminator, Generator


class WGANGPInference:
    def __init__(self, generator_path="", discriminator_path="", channels_img=3, features_gen=64, features_disc=64,
                 noise_dim=100, device=None):
        """
        WGANGP参数解释：
        generator_path: 推理时使用的生成器模型路径
        discriminator_path: 推理时使用的判别器模型路径
        channels_img: 图像通道数
        features_gen: 生成器的特征数
        features_disc: 判别器的特征数
        noise_dim: 生成器的输入噪声维度
        device: 推理设备
        """
        self.device = device if device else torch.device('cpu')
        _current_file_path = os.path.abspath(__file__)
        _current_directory = os.path.dirname(_current_file_path)
        if len(generator_path) == 0:
            self.generator_path = os.path.join(_current_directory, "models/gen.pth")
        else:
            self.generator_path = generator_path
        if len(discriminator_path) == 0:
            self.discriminator_path = os.path.join(_current_directory, "models/disc.pth")
        else:
            self.discriminator_path = discriminator_path

        # 加载生成器和判别器
        self.generator = Generator.Generator(noise_dim, channels_img, features_gen).to(self.device)
        self.discriminator = Discriminator.Discriminator(channels_img, features_disc).to(self.device)

        self.generator.load_state_dict(torch.load(self.generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(self.discriminator_path, map_location=self.device))

        self.generator.eval()
        self.discriminator.eval()

    @torch.no_grad()
    def generate_images(self, max_images, final_images, noise_dim=100, seed=-1):
        """
        生成图片并筛选。
        :param max_images: 最大生成的图像数量
        :param final_images: 最终输出的图像数量
        :param noise_dim: 随机噪声的维度
        :param seed: 随机噪声种子
        :return: 最终筛选的图像列表 (Tensor 格式)
        """
        if seed != -1:
            torch.manual_seed(seed)

        # 生成随机噪声
        noise = torch.randn(max_images, noise_dim, 1, 1, device=self.device)

        # 用生成器生成图片
        fake_images = self.generator(noise)

        # 使用判别器对图片打分
        scores = self.discriminator(fake_images).reshape(-1)

        # 按判别器的分数筛选前 `final_images` 张
        top_indices = torch.topk(scores, k=min(final_images, max_images)).indices
        selected_images = fake_images[top_indices]

        return selected_images


def generate(generator_path="", discriminator_path="", channels_img=3, features_gen=64, features_disc=64, noise_dim=100,
             device=None,
             max_images=100, final_images=10, seed=-1, output_dir="./generated_images"):
    """
    生成并保存图像。
    :param generator_path: 推理时使用的生成器模型路径
    :param discriminator_path: 推理时使用的判别器模型路径
    :param channels_img: 图像通道数
    :param features_gen: 生成器的特征数
    :param features_disc: 判别器的特征数
    :param noise_dim: 生成器的输入噪声维度
    :param device: 推理设备
    :param max_images: 最大生成的图像数量
    :param final_images: 最终输出的图像数量
    :param seed: 随机噪声种子
    :param output_dir: 保存生成图像的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 实例化推理类
    inference = WGANGPInference(generator_path=generator_path, discriminator_path=discriminator_path,
                                channels_img=channels_img, features_gen=features_gen, features_disc=features_disc,
                                noise_dim=noise_dim, device=device)

    # 生成图像
    final_images_tensor = inference.generate_images(
        max_images=max_images,
        final_images=final_images,
        noise_dim=noise_dim,
        seed=seed
    )

    # 保存生成的图像
    for idx, image in enumerate(final_images_tensor):
        output_path = os.path.join(output_dir, f"generated_image_{idx + 1}.png")
        torchvision.utils.save_image(image, output_path, normalize=True)
        print(f"Saved image to {output_path}")


if __name__ == "__main__":
    generate(
        generator_path=r"./training/WGANGP/models/gen.pth",
        discriminator_path=r"./training/WGANGP/models/disc.pth",
        channels_img=3,
        features_gen=64,
        features_disc=64,
        noise_dim=100,
        device=None,
        max_images=100,
        final_images=10,
        seed=42,
        output_dir="./generated_images"
    )
