import os
from ddpm import Diffusion

#-------------------------------------#
#   运行predict.py可以生成图片
#   生成1x1的图片
#-------------------------------------#
def predict_demo( save_dir="results/predict_out"):
    ddpm = Diffusion()

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for i in range(5):
        save_path_1x1 = os.path.join(save_dir, f"predict_1x1_result_{i}.png")
        print(f"Generate 1x1 image {i + 1}...")
        ddpm.generate_1x1_image(save_path_1x1)
        print(f"Generate 1x1 image {i + 1} Done. Saved to: {save_path_1x1}")

if __name__ == "__main__":

    predict_demo()

