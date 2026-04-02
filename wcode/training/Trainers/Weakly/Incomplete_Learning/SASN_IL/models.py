import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from wcode.net.CNN.VNet.VNet import VNet


class MeanTeacher(nn.Module):
    def __init__(self, params, alpha=0.99):
        super(MeanTeacher, self).__init__()
        self.student = VNet(params)
        self.teacher = VNet(params)

        # initialize the params of teacher the same as the student's.
        self.init_teacher_with_student()
        self.teacher.requires_grad_(False)
        self.check_init_weight()

        self.alpha = alpha

    def forward(self, x, train_flag=False, weak_aug=False):
        if train_flag:
            if weak_aug:
                with torch.no_grad():
                    rotate_time = random.randint(1, 3)
                    flip_axis = [np.random.choice(list(range(x.ndim))[-2:])]
                    x_s = x.clone().detach()
                    x_s = self.rotate_img(x_s, rotate_time)
                    x_s = self.flip_img(x_s, flip_axis)

                    if torch.rand(1) < 0.5:
                        x_s = self.gaussian_blur(x_s)
                    else:
                        x_s = self.sharpen(x_s)

                student_out = self.student(x_s)

                student_out["pred"] = self.flip_img(student_out["pred"], flip_axis)
                student_out["pred"] = self.rotate_img(
                    student_out["pred"], 4 - rotate_time
                )
                if "feature" in student_out.keys():
                    student_out["feature"] = self.flip_img(
                        student_out["feature"], flip_axis
                    )
                    student_out["feature"] = self.rotate_img(
                        student_out["feature"], 4 - rotate_time
                    )
            else:
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

    def exchange_params_teacher_and_student(self):
        params = self.student.state_dict().copy()
        self.student.load_state_dict(self.teacher.state_dict())
        self.teacher.load_state_dict(params)

    def check_init_weight(self):
        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            assert torch.equal(student_param.data, teacher_param.data)

    def rotate_img(self, img, rotate_time):
        if isinstance(img, (tuple, list)):
            dim = len(img[0].shape)
            return [
                torch.rot90(i, k=rotate_time, dims=(2, 3) if dim == 4 else (3, 4))
                for i in img
            ]
        else:
            dim = len(img.shape)
            return torch.rot90(img, k=rotate_time, dims=(2, 3) if dim == 4 else (3, 4))

    def flip_img(self, img, filp_axis):
        if isinstance(img, (tuple, list)):
            return [torch.flip(i, dims=filp_axis) for i in img]
        else:
            return torch.flip(img, dims=filp_axis)

    def gaussian_blur(self, img):
        dim = img.ndim - 2

        def gaussian_kernel(dim, kernel_size=3, sigma=2.0):
            if isinstance(kernel_size, int):
                size = [int(kernel_size) // 2 for _ in range(2)]
            elif isinstance(kernel_size, (list, tuple)):
                size = [i // 2 for i in list(kernel_size)]
            else:
                raise ValueError("Unsupport type of kernel_size:", type(kernel_size))

            x, y = np.mgrid[-size[0] : size[0] + 1, -size[1] : size[1] + 1]
            g = np.exp(-(x**2 + y**2) / (2 * sigma**2))

            return g / g.sum(), size + size

        if dim == 3:
            # b, c, z, y, x -> b, z, c, y, x -> b*z, c, y, x
            b, c, z, y, x = img.shape
            img = img.transpose(1, 2).reshape(b * z, c, y, x)
        elif dim == 2:
            b, c, y, x = img.shape

        kernelsize = random.randint(3, 7)
        kernel, padding_size = gaussian_kernel(dim, kernel_size=kernelsize)
        kernel_tensor = (
            torch.from_numpy(kernel)
            .float()
            .to(img.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(c, 1, 1, 1)
        )  # (c, 1, kernel_size, kernel_size)

        blurred_img = F.conv2d(
            F.pad(img, pad=padding_size, mode="reflect"), kernel_tensor, groups=c
        )

        if dim == 3:
            blurred_img = blurred_img.reshape(b, z, c, y, x).transpose(1, 2)

        return blurred_img

    def sharpen(self, img):
        dim = img.ndim - 2

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        if dim == 3:
            # b, c, z, y, x -> b, z, c, y, x -> b*z, c, y, x
            b, c, z, y, x = img.shape
            img = img.transpose(1, 2).reshape(b * z, c, y, x)
        elif dim == 2:
            b, c, y, x = img.shape

        kernel_tensor = torch.from_numpy(sharpen_kernel).float().to(img.device)
        kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)

        sharpened_img = F.conv2d(
            F.pad(img, pad=[1, 1, 1, 1], mode="reflect"), kernel_tensor, groups=c
        )

        if dim == 3:
            sharpened_img = sharpened_img.reshape(b, z, c, y, x).transpose(1, 2)

        return sharpened_img
