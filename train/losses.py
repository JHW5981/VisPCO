import torch
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
# from calc_flops_utils import custom_ops

def create_criterion(teacher_model, model, args):
    criterion = DynamicPruneDistillLoss(
        teacher_model=teacher_model,
        model=model,
        print_mode=args.print_mode,
        args=args
    )
    return criterion

class DynamicPruneDistillLoss(nn.Module):
    def __init__(self, 
                 teacher_model, 
                 model, 
                 cls_weight=1.0, 
                 print_mode=True,
                 args=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.model = model
        self.cls_weight = cls_weight
        self.print_mode = print_mode
        self.count = 0
        self.distill_label_loss = 0.
        self.flops_loss = 0.
        self.cls_loss = 0.

        # 拉格朗日乘子参数
        self.w = args.w
        self.sigma = args.sigma
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.beta = args.beta

    def forward(self, outputs, batch):
        # target_model = self.model.module if hasattr(self.model, 'module') else self.model

        loss_part = {}

        # 第一个loss是蒸馏loss，teacher model的label部分和student model的label部分
        student_label_logits = outputs.label_logits
        with torch.no_grad():
            teacher_label_logits = self.teacher_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                position_ids=batch['position_ids'],
                image_grid_thw=batch['image_grid_thw'],
                labels=batch['labels']
            ).label_logits
        assert student_label_logits.shape == teacher_label_logits.shape

        distill_label_loss = F.kl_div(
                F.log_softmax(student_label_logits, dim=-1),
                F.log_softmax(teacher_label_logits, dim=-1),
                reduction='batchmean',
                log_target=True
            ) 
        # 第二个loss是FLOPs的约束
        flops_loss = outputs.flop_loss

        # 拉格朗日损失函数
        # z = torch.clamp(self.w - self.sigma * flops_loss, min=0)
        # langrange_loss = 1000 * distill_label_loss + 1 / (2 * self.sigma) * (z**2 - self.w**2)
        langrange_loss = 1000 * distill_label_loss - 0.1 * self.w * flops_loss + self.sigma * flops_loss**2
        loss = langrange_loss
        # loss = self.flops_weight * flops_loss + self.distill_weight * distill_label_loss
        # loss = self.flops_weight * flops_loss

        if self.print_mode:
            self.distill_label_loss += distill_label_loss.item()
            self.flops_loss += flops_loss.item()
            loss_part['distill_label_loss'] = distill_label_loss
            loss_part['flops_loss'] = flops_loss
            self.count += 1
            if self.count % 100 == 0:
                print('loss info: distill_label_loss=%.4f, flops_loss=%.4f' % (self.distill_label_loss / 100, self.flops_loss / 100))
                self.distill_label_loss = 0.
                self.flops_loss = 0.

        return loss, loss_part


