import os
import time
import torch
import shutil
import logging
import argparse
from architectures import UNet2D
from losses import DiceLoss, BCE
import cli, ramps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = logging.getLogger('main')

args = None
best_pre = 0
global_step = 0

def main():
    global best_pre
    global global_step

    student = create_model()
    teacher = create_model(ema=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        start_time = time.time()
        train(train_loader, student, teacher, optimizer, epoch)
        LOG.info("Training epoch in %s seconds" % (time.time()-start_time))

        LOG.info("Evaluating the student model")
        student_pre = validate(eval_loader, student)

        LOG.info("Evaluating the teacher model")
        teacher_pre = validate(eval_loader, teacher)

        if teacher_pre > best_pre:
            best_pre = max(teacher_pre, best_pre)
            save_best_checkpoint({'epoch': epoch + 1, 'state_dict': student.state_dict(), 'ema_state_dict': teacher.state_dict(), 
            'best_pre': best_pre, 'optimizer' : optimizer.state_dict(), }, args.checkpoint_path, epoch + 1)


def save_best_checkpoint(state, dirpath, epoch):
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, best_path)
    LOG.info("Best checkpoint saved to %s" % best_path)


def create_model(ema=False):
        LOG.info("=> creating {ema}model".format(ema='EMA ' if ema else ''))
        model = UNet2D().to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, student, teacher, optimizer, epoch):
    segmentation_criterion = DiceLoss()
    consistency_criterion = BCE()

    #switch to train mode
    student.train()
    teacher.train()
    
    for i ((stu_input, tea_input), target) in enumerate(train_loader):
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    
    