import torch.nn as nn
import torch
# from objectives import cca_loss

train_num = 2000
k_num = 10
in_size = 28
out_size = 7
out_num = 1470


class Networks(nn.Module):
    def __init__(self):
        super(Networks, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(20, 30, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 1, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.model1 = nn.Linear(out_num, k_num)
        self.model2 = nn.Linear(out_num, k_num)
        # self.loss = cca_loss(k_num, torch.device('cuda')).loss
        self.weight = nn.Parameter(1.0e-4 * torch.ones(train_num, train_num))

    def forward(self, input1, input2):
        coef = self.weight - torch.diag(torch.diag(self.weight))

        z1 = self.encoder1(input1)
        z1 = z1.view(train_num, out_num)
        zcoef1 = torch.matmul(coef, z1)
        output1 = zcoef1.view(train_num, 30, out_size, out_size)
        output1 = self.decoder1(output1)

        z2 = self.encoder2(input2)
        z2 = z2.view(train_num, out_num)
        zcoef2 = torch.matmul(coef, z2)
        output2 = zcoef2.view(train_num, 30, out_size, out_size)
        output2 = self.decoder2(output2)

        return output1, output2, zcoef1, zcoef2, coef, z1, z2

    def forward2(self, input1, input2):
        coef = self.weight - torch.diag(torch.diag(self.weight))

        z1 = self.encoder1(input1)
        z1 = z1.view(train_num, out_num)
        z11 = self.model1(z1)
        zcoef1 = torch.matmul(coef, z1)
        output1 = zcoef1.view(train_num, 30, out_size, out_size)
        output1 = self.decoder1(output1)

        z2 = self.encoder2(input2)
        z2 = z2.view(train_num, out_num)
        z22 = self.model2(z2)
        zcoef2 = torch.matmul(coef, z2)
        output2 = zcoef2.view(train_num, 30, out_size, out_size)
        output2 = self.decoder2(output2)

        return z11, z22, z1, z2, output1, output2, zcoef1, zcoef2, coef
