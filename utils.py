import torch
from losses import DiceLoss
from data import OriSet
from torch.utils.data import DataLoader
import numpy as np

class CaseTester():
    
    def __init__(self, model, name, device, test_cases):
        self.model = model
        self.device = device
        self.test_cases = test_cases
        self.test_result = []
        self.name = name
        

    def get_loader(self, case_id):
        case_path = "dataset/case_data/coronacases_" + str(case_id).zfill(3)
        case_set = OriSet(case_path)
        case_loader = DataLoader(case_set, 1, num_workers=0, pin_memory=False, shuffle=True)
        return case_loader
        
    def case_test(self, model, device, test_loader, case_id):
        self.model.eval()
        testNum = len(test_loader.dataset)
        diceloss = 0.
        Criterion = DiceLoss()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data).view_as(target)
                diceloss += Criterion(output, target)

        diceloss /= testNum
        dice = 1 - diceloss
        return dice.item()

    def run_test(self):
        for case_id in self.test_cases:
            case_loader = self.get_loader(case_id)
            case_dice = self.case_test(self.model, self.device, case_loader, case_id)
            self.test_result.append(case_dice)
        aveDice = np.array(self.test_result).mean()
        self.test_result = [] # 清空test_result
        print("The average test Dice of {} is {:.4f}".format(self.name, aveDice))
        return aveDice