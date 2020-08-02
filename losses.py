import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, ignore=False, smooth=1, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore = ignore

    def forward(self, preds, labels):
        if self.ignore:
            preds = preds.squeeze()
            labels = labels.squeeze()

            ignored_index = (labels != -1).int()

            preds = preds * ignored_index
            labels = labels * ignored_index

        return 1 - (2 * torch.sum(preds * labels) + self.smooth) / (torch.sum(preds) + torch.sum(labels) + self.smooth)


class BCE(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, student_output, teacher_output):
        loss = -(teacher_output * torch.log(student_output + self.eps) + (1-teacher_output) * torch.log(1-student_output + self.eps))
        return loss.sum() / student_output.nelement()


        

        