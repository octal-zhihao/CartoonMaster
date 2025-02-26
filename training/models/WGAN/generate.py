import yaml
from yacs.config import CfgNode

from training.models.WGAN.utils import *


def main(cfg):
    gen_num = cfg['gen_num']
    # print(gen_num)
    cfg["IMAGE"]["SAVE_NUMBER"] = int(gen_num)
    # print(cfg["IMAGE"]["SAVE_ROW_NUMBER"])
    # print(cfg)
    cfg = CfgNode(cfg)
    cfg.SOLVER.BETAS = [float(b) for b in cfg.SOLVER.BETAS.split(',')]
    # cfg = get_cfg()
    # print(type(cfg))
    # print(cfg)
    # cfg = project_preprocess(cfg)
    # print(type(cfg))
    # print(cfg)
    model = build_model(cfg)
    model.load_model()
    model.generate_images()
    print('Generate images successfully!')
    

if __name__ == '__main__':
    with open('../../config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['model']['WGAN']
    main(cfg)
