import os

from options import *
from draft import DraftDrawer
from refine import Refiner

def get_model(opt):
    if opt.model == 'draft':
        model = DraftDrawer(opt)
    elif opt.model == 'refine':
        model = Refiner(opt)
    else:
        raise NotImplementedError('Such model is not implemented.')
    return model


if __name__ == '__main__':
    opt = get_options()

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    model = get_model(opt)

    if not opt.eval:
        model.train()
    else:
        model.test()
