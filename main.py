import os
import time
import torch
import shutil
import logging
import argparse
from architectures import UNet2D
from losses import DiceLoss
import torch.nn as nn
from data import TwoStreamBatchSampler, SemiSet
import cli, ramps
from utils import CaseTester

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = logging.getLogger('main')

args = None
best_dice = 0
global_step = 0

def main():
    global best_dice
    global global_step
    test_cases = range(240, 290)

    teacher = create_model(ema=True)
    student = create_model()

    # get models
    if args.self_supervised:
        teacher.load_state_dict(torch.load('model/snr=6.ckpt'))
        student.load_state_dict(torch.load('model/snr=6.ckpt'))

    # get dataLoader and optimizer
    train_loader = build_loader(args.dataset_folder)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # use consine annealing to update learning rate
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 4)


    # get caseTester for two models
    student_tester = CaseTester(student, 'student', device, test_cases)
    teacher_tester = CaseTester(teacher, 'teacher', device, test_cases)

    for epoch in range(args.epochs):

        start_time = time.time()
        train(train_loader, student, teacher, optimizer, epoch)
        LOG.info("Training epoch in %s seconds" % (time.time()-start_time))

        if args.cosine_annealing:
            scheduler.step()

        LOG.info("Evaluating the student model")
        student_dice = student_tester.run_test()

        LOG.info("Evaluating the teacher model")
        teacher_dice = teacher_tester.run_test()

        if teacher_dice > best_dice:
            best_dice = max(teacher_dice, best_dice)
            LOG.info("Current best dice is changed to {:.3f}".format(best_dice))
            save_best_checkpoint({'epoch': epoch + 1, 'state_dict': student.state_dict(), 'ema_state_dict': teacher.state_dict(), 
            'best_pre': best_dice, 'optimizer' : optimizer.state_dict(), }, args.checkpoint_path, epoch + 1)
        
        print("------------------------")
            
    LOG.info("The best dice is {:.3f}".format(best_dice))
    


def save_best_checkpoint(state, dirpath, epoch):
    best_path = os.path.join(dirpath, args.best_checkpoint_name)
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


def build_loader(dataset_folder):

    data_dir = os.path.join(dataset_folder, "data")
    lab_dir = os.path.join(dataset_folder, "label")

    data_num = len(os.listdir(data_dir))
    lab_num = len(os.listdir(lab_dir))

    unlabeled_idx = range(data_num - lab_num)
    labeled_idx = range(data_num - lab_num, data_num)

    sampler = TwoStreamBatchSampler(unlabeled_idx, labeled_idx, args.batch_size, args.labeled_batch_size)
    dataset = SemiSet(dataset_folder, args.augmentation)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    return loader
    


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, student, teacher, optimizer, epoch):
    global global_step
    global best_dice

    # set criterion
    segmentation_criterion = DiceLoss(ignore=True)
    consistency_criterion = nn.MSELoss()

    #switch to train mode
    student.train()
    teacher.train()
    
    for i, (data, label) in enumerate(train_loader):
        # if we don't use cosine_annealing, adjust learning rate
        if args.cosine_annealing == False:
            adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        data = data.to(device)
        label = label.to(device)

        # get the result of the two models
        student_pred = student(data)
        teacher_pred = teacher(data)
        # We don't want gradient descent in teacher model.
        teacher_pred = torch.autograd.Variable(teacher_pred.detach().data, requires_grad=False)

        # calculate consistency criterion
        consistency_weight = get_current_consistency_weight(epoch)
        consistency_loss = consistency_weight * consistency_criterion(student_pred, teacher_pred)
        # calculate segmentation loss
        segmentation_loss = segmentation_criterion(student_pred, label)
        # combine them to get the final loss
        loss = consistency_loss + segmentation_loss

        # compute gradient and do optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(student, teacher, args.ema_decay, global_step)

        # print training information
        if i % args.print_freq == 0:
            LOG.info('Epoch: [{}][{}/{}]\tLoss: {:.3f}\t'.format(epoch, i, len(train_loader), loss))
            LOG.info('Conssistency Loss: {:.3f}\t'.format(consistency_loss))



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main()


    


    


    

    
    
    