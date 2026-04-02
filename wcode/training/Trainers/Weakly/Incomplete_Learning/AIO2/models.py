import torch
import torch.nn as nn

from wcode.net.CNN.VNet.VNet import VNet


class MeanTeacher(nn.Module):
    def __init__(self, params, alpha=0.999):
        super(MeanTeacher, self).__init__()
        self.student = VNet(params)
        self.teacher = VNet(params)

        # initialize the params of teacher the same as the student's.
        self.init_teacher_with_student()
        self.teacher.requires_grad_(False)

        self.check_init_weight()

        self.alpha = alpha

    def forward(self, x, train_flag=False):
        if train_flag:
            student_out = self.student(x)
        else:
            student_out = None

        with torch.no_grad():
            teacher_out = self.teacher(x)

        return {
            "student_out": student_out,
            "teacher_out": teacher_out,
            "pred": teacher_out["pred"],
        }

    def update_ema_variables(self, alpha=None):
        if alpha is None:
            alpha = self.alpha

        for ema_param, param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def update_alpha(self, new_value):
        assert 0 <= new_value <= 1
        self.alpha = new_value

    def init_teacher_with_student(self):
        self.teacher.load_state_dict(self.student.state_dict())

    def check_init_weight(self):
        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            assert torch.equal(student_param.data, teacher_param.data)
