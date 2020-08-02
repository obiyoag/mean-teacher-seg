import re
import argparse
import logging


LOG = logging.getLogger('main')


def create_parser():
    parser = argparse.ArgumentParser(description='Mean-teacher Method for COVID-19')
    parser.add_argument('--dataset-folder', type=str, default='dataset', help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--labeled-batch-size', default=8, type=int, metavar='N', help='num of labeled data in a mini-batch')
    parser.add_argument('--lr', default=3e-4, type=float, metavar='LR', help='max learning rate')
    parser.add_argument('--cosine-annealing', default=False, type=bool, help='whether to use cosine annealing to update learning rate')
    parser.add_argument('--initial-lr', default=1e-4, type=float, metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=50, type=int, metavar='EPOCHS', help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS', help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA', help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=200.0, type=float, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=150, type=int, metavar='EPOCHS', help='length of the consistency loss ramp-up')
    parser.add_argument('--checkpoint_path', default="model", type=str, help='the directory to save the best checkpoint')
    parser.add_argument('--best-checkpoint-name', default='best.ckpt', type=str, help='the name of the best checkpoint to be saved')
    parser.add_argument('--augmentation', default=False, type=bool, help='whether to use data augmentation')
    parser.add_argument('--print-freq', '-p', default=300, type=int, metavar='N', help='print frequency (default: 500)')
    parser.add_argument('--self-supervised', default=False, type=bool, help='whether to use self-supervised pretrain model')
    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value) for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError('Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError('Expected the epochs to be listed in increasing order')
    return epochs