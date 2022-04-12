from options import Options
from draft import DraftDrawer
from mapping import Mapper

if __name__ == '__main__':
    parser = Options(eval=False)
    opt = parser.get_options()
    # opt.attn_type = opt.attn_type if opt.model == 'colorization' else 'add'

    if opt.model == 'mapping':
        opt.dataset_mode = 'mapping'
        model = Mapper(opt)
    else:
        opt.dataset_mode = 'colorization'
        model = DraftDrawer(opt)
    opt.data_size = model.data_size
    parser.print_options(opt, phase='train')
    model.train_loop()
