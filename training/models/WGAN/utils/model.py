from training.models.WGAN.model import *

def build_model(cfg):

    total_model = {
        'WGAN': WGAN,
    }
    model = total_model[cfg.MODEL.NAME](cfg)

    if cfg.MODEL.DEVICE == 'cuda':
        model = model.cuda()
    
    return model