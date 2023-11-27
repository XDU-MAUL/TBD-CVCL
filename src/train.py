import torch
from data.dataloader import data_loader_train
from models.network import Networks
import models.metrics as metrics
import numpy as np
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from models import contrastive_loss, newcontrast
import torch.nn.functional as F

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


train_num = 2000
k_num = 10
in_size = 28

data_0 = sio.loadmat('Dataset/fashion_MNIST_2000.mat')
data_dict = dict(data_0)
label_true = np.squeeze(np.array(data_dict['groundtruth'].T).astype(int))
print(label_true)

reg2 = 1.0 * 10 ** (k_num / 10.0 - 3.0)

model = Networks().cuda()
# model.load_state_dict(torch.load('./models/50-AE1.pth'))
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.0)
n_epochs = 100
for epoch in range(n_epochs):
    for data in data_loader_train:
        train_imga, train_imgb = data
        input1 = train_imga.view(train_num, 1, in_size, in_size).cuda()
        input2 = train_imgb.view(train_num, 1, in_size, in_size).cuda()
        output1, output2, zcoef1, zcoef2, coef, z1, z2 = model(input1, input2)

        loss_r = 0.5 * criterion(output1, input1) + 0.5 * criterion(output2, input2)  # 预损失
        loss_self = criterion(coef, torch.zeros(train_num, train_num, requires_grad=True).cuda())
        loss_e = 0.5 * criterion(zcoef1, z1) + 0.5 * criterion(zcoef2, z2)
        pre_loss = loss_r + loss_self + loss_e
        optimizer.zero_grad()
        pre_loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs))
        print("Pre_Loss is:{:.4f}".format(pre_loss.item()))
        if epoch % 20 == 0:
            commonZ = coef.cpu().detach().numpy()
            alpha = max(0.4 - (k_num - 1) / 10 * 0.1, 0.1)
            commonZ = thrC(commonZ, alpha)
            preds, _ = post_proC(commonZ, k_num)
            acc = metrics.acc(label_true, preds)
            nmi = metrics.nmi(label_true, preds)
            print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                  % (acc, nmi))
# sio.savemat('pre_loss' + '.mat', {'pre_loss': pre_loss})
# torch.save(model.state_dict(), './models/50-AE1.pth')

'''



'''

print("step2")
print("---------------------------------------")
criterion2 = torch.nn.MSELoss(reduction='sum')

# Define contrastive_loss
instance_temperature = 0.1
loss_device = torch.device("cuda")
# criterion_instance = contrastive_loss.InstanceLoss(train_num, instance_temperature, loss_device)
instance_loss = newcontrast.InstanceLoss(train_num, instance_temperature, loss_device)


optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.0)
n_epochs2 = 1000
ACC_MNIST = []
NMI_MNIST = []
Loss_re = []
Loss_cl = []
Loss_r = []
Loss = []
maxacc = 0
iter = 0
for epoch in range(n_epochs2):
    for data in data_loader_train:
        train_imga, train_imgb = data
        input1 = train_imga.view(train_num, 1, in_size, in_size).cuda()
        input2 = train_imgb.view(train_num, 1, in_size, in_size).cuda()
        z11, z22, z1, z2, output1, output2, zcoef1, zcoef2, coef = model.forward2(input1, input2)
        # coef = model.weight - torch.diag(torch.diag(model.weight))
        # commonZ = coef.cpu().detach().numpy()
        z11=F.normalize(z11, dim=-1)
        z22=F.normalize(z22, dim=-1)
        A = (coef + coef.t()) / 2
        loss_instance = instance_loss(z11,z22,A,k=200)

        loss_re = criterion2(coef, torch.zeros(train_num, train_num, requires_grad=True).cuda())
        loss_e = 0.5 * criterion2(zcoef1, z1) + 0.5 * criterion2(zcoef2, z2)
        loss_r = 0.5 * criterion2(output1, input1) + 0.5 * criterion2(output2, input2)
        loss = 1 * loss_r + 1 * (loss_re + 1 * reg2 * loss_e)  + 1* loss_instance
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    if epoch % 1 == 0:
        print("Epoch {}/{}".format(epoch, n_epochs2))
        print("Loss_re is:{:.4f}".format(loss_re.item()))
        print("Loss_r is:{:.4f}".format(loss_r.item()))
        print("Loss is:{:.4f}".format(loss.item()))
        print("contrastive_loss is:{:.4f}".format(loss_instance.item()))
        coef = model.weight - torch.diag(torch.diag(model.weight))
        commonZ = coef.cpu().detach().numpy()
        alpha = max(0.4 - (k_num - 1) / 10 * 0.1, 0.1)
        commonZ = thrC(commonZ, alpha)
        preds, _ = post_proC(commonZ, k_num)
        acc = metrics.acc(label_true, preds)
        nmi = metrics.nmi(label_true, preds)
        ACC_MNIST.append(acc)
        NMI_MNIST.append(nmi)
        Loss_re.append(loss_re.item())
        Loss_r.append(loss_r.item())
        Loss_cl.append(loss_instance.item())
        Loss.append(loss.item())
        print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
              % (acc, nmi))
        # if acc > maxacc:
        #     maxacc = acc
        #     iter = epoch
        #     Z_path = 'commonZ' + str(epoch)
        #     sio.savemat(Z_path + '.mat', {'Z': commonZ, 'Pred': preds})

        #     print(str(iter))

        #     sio.savemat('Pred' + str(epoch) + '.mat', {'Pred': preds})
        #     Z_path = 'commonZ' + str(epoch)
        #     sio.savemat(Z_path + '.mat', {'Z': commonZ})
        # sio.savemat('ACC_MNIST' + '.mat', {'acc_mnist': ACC_MNIST})
        # sio.savemat('NMI_MNIST' + '.mat', {'nmi_mnist': NMI_MNIST})
        sio.savemat('evaluation' + '.mat', {'nmi_mnist': NMI_MNIST, 'acc_mnist': ACC_MNIST,
                                            'Loss': Loss, 'Loss_re': Loss_re, 'Loss_r': Loss_r, 'Loss_cl': Loss_cl})
        # sio.savemat('Loss_re' + '.mat', {'Loss_re': Loss_re})
        # sio.savemat('Loss_r' + '.mat', {'Loss_r': Loss_r})
        # sio.savemat('Loss' + '.mat', {'Loss': Loss, 'Loss_re': Loss_re, 'Loss_r': Loss_r, 'Loss_cl': Loss_cl})
# torch.save(model.state_dict(), './models/AE2.pth')
