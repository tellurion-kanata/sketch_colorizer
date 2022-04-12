from options import Options
from draft import DraftDrawer
from mapping import Mapper

if __name__ == '__main__':
    parser = Options(eval=True)
    opt = parser.get_options()

    opt.dataset_mode = 'colorization'
    opt.no_ref_shuffle = True
    if opt.save_input:
        opt.batch_size = 1

    if opt.model == 'mapping':
        model = Mapper(opt)
    else:
        model = DraftDrawer(opt)
    opt.data_size = model.data_size
    parser.print_options(opt, phase='test')
    model.test(opt.load_epoch)
