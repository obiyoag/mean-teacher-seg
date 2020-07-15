import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, labels):
        return 1 - (2 * torch.sum(preds * labels) + self.smooth) / (torch.sum(preds) + torch.sum(labels) + self.smooth)


class BCE(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, student_output, teacher_output):
        loss = -(teacher_output * torch.log(student_output + self.eps) + (1-teacher_output) * torch.log(1-student_output + self.eps))
        print(len(student_output))
        return loss.sum() / len(student_output)
        

bceloss = BCE()
student_output = torch.tensor([[[1,1,0.99],[0,0,0],[0,0,0]],[[1,1,0.99],[0,0,0],[0,0,0]]])
teacher_output = torch.tensor([[[1,1,0.99],[0,0,0],[0,0,0]],[[1,1,0.99],[0,0,0],[0,0,0]]])

loss = bceloss(student_output, teacher_output)
print(loss)
        