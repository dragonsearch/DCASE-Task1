from core.train.trainer_mixup import TrainerMixUp
import torch
import torch as torch
import torch.nn as nn
from tools.utils import load_ckpt


class TrainerKD(TrainerMixUp):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.teacher_name = params['teacher_name']
        # teacher name = TFSEPNET_less_relu_80_nomixup_lr[0.25, 0.25, 0.1, 0, 0.0, 0.2, 0.2]_0
        # epoch =199
        
        self.temperature = params['temperature']
        self.weight_teacher = params['weight_teacher']
        self.load_teacher_model(params['teacher_epoch'])

    def train_step(self, samples, labels_a, labels_b, lam):
        # Forward pass
        self.optimizer.zero_grad()
        y_pred = self.model(samples)
        # Forward pass for teacher model
        # We don't need to compute the gradients for the teacher model
        with torch.no_grad():
            self.teacher_model.eval()
            y_teacher = self.teacher_model(samples)
            self.teacher_model.train()
        """
        CE
        """
        """
        soft_teacher = nn.functional.softmax(y_teacher / self.temperature, dim=-1)
        # (Not applicable when using CE but Input KL divergence must be log_softmax)
        soft_prob_student = nn.functional.log_softmax(y_pred / self.temperature, dim=-1)
        # Compute the loss Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_target_loss = torch.sum(soft_teacher* (soft_teacher.log() - soft_prob_student)) / soft_prob_student.size()[0]*self.temperature**2
        """
        """
        KL divergence
        """
        
        
        soft_teacher = nn.functional.log_softmax(y_teacher / self.temperature, dim=-1)
        soft_prob_student = nn.functional.log_softmax(y_pred / self.temperature, dim=-1)
        kl_loss = nn.functional.kl_div(soft_prob_student, soft_teacher, reduction='batchmean', log_target=True)

        soft_target_loss = kl_loss * self.temperature**2
        
        loss_student = self.mixUpCriterion(y_pred, labels_a, labels_b, lam)
        """ 
        CE
        """
        #loss_student = self.weight_teacher * soft_target_loss + (1 - self.weight_teacher) * loss_student
        # Perform weighted sum of the losses using lerp
        loss = torch.lerp(soft_target_loss, loss_student, self.weight_teacher)
        """
        KL
        """
        #loss = loss_student + self.weight_teacher * soft_target_loss
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        return y_pred, loss
    
    def load_teacher_model(self, epoch):
        ckpt_path = 'models/' + self.teacher_name + "/ckpt" + "/model_" + str(self.teacher_name) + '_' + str(epoch) + ".pth"
        lr_sc_dummy = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        optimizer_dummy = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        from model_classes.tfsepnet import TfSepNet
        teacher_model = TfSepNet(width=80, shuffle_groups=10)
        #self.teacher_model, self.optimizer = load_ckpt(self.model, self.optimizer, self.lr_scheduler, ckpt_path)
        self.teacher_model, _ = load_ckpt(teacher_model, optimizer_dummy, lr_sc_dummy, ckpt_path)
        self.teacher_model.to(self.device)
        print(type(self.teacher_model))
        print("Loading model with loss: ", self.loss_dict["train"][epoch], "from ", ckpt_path)
