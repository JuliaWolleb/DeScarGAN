# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import torchvision
from visdom import Visdom
import matplotlib.pyplot as plt
import os
viz=Visdom(port=8850)
from utils.tools import  npy_loader,   normalize,  kappa_score
import torch.nn as nn
from utils.Functions import  create_labels
from model.generator_discrminator import Generator, Discriminator
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

id=0
torch.cuda.set_device(id)
device='cuda'
print("computations done on ", device)
class Solver(object):
    def __init__(self,dataset_path, choose_net):
        super(Solver, self).__init__()
        self.dataset_path=dataset_path
        self.choose_net=choose_net



    def test(self):
        path = os.path.join(self.dataset_path, 'test')
        print(path, 'PATH')

        path2 = self.choose_net

        netG=Generator().to(device)
        netD=Discriminator().to(device)

        p1=np.array([np.array(p.shape).prod() for p in netG.parameters()]).sum()
        p2=np.array([np.array(p.shape).prod() for p in netD.parameters()]).sum()
        print(p1, p2, p1/(256*256))

        beta1=0.5
        beta2=0.999

        Testset = torchvision.datasets.DatasetFolder(
            root=path,
            loader=npy_loader,
            extensions=('.npy',)
        )

        test_loader = torch.utils.data.DataLoader(dataset=Testset,
                                                   batch_size=1,
                                                   shuffle=False)




        g_optimizer = torch.optim.Adam(netG.parameters(), 0.0001, [beta1, beta2])
        d_optimizer = torch.optim.Adam(netD.parameters(), 0.0001, [beta1, beta2])
        netG, g_optimizer = amp.initialize(netG, g_optimizer, opt_level='O1')
        netD, d_optimizer = amp.initialize(netD, d_optimizer, opt_level='O1')

        netG = torch.nn.DataParallel(netG, device_ids=[id], output_device=id)
        netD = torch.nn.DataParallel(netD, device_ids=[id], output_device=id)

        g=0

        try_loading_file =True
        if try_loading_file:
            try:
                netG.load_state_dict(torch.load(os.path.join(path2, 'netG_chexpert.pt'),map_location={'cuda:0': 'cpu'})) #does NOT load optimizer state etc.
                netD.load_state_dict(torch.load(os.path.join(path2, 'netD_chexpert.pt'),map_location={'cuda:0': 'cpu'})) #does NOT load optimizer state etc.

                print("loaded model from file")
            except:
                print("loading model from file failed; created new model")

        c_dim=2
        # =================================================================================== #
        #                                 5. Testing                                          #
        # =================================================================================== #



        """Translate images using StarGAN trained on a single dataset."""
        netD = netD.eval()
        total=0; correct=0; count_gesund=0; total_rec=0; total_var=0;
        loss_metric = nn.MSELoss()
        with torch.no_grad():
            long_pred = torch.zeros(0).long()
            long_cls = torch.zeros(0).long()
            long_score = torch.zeros(0)
            for i, (X, label_org) in enumerate(test_loader):
              if i<500:

                # Prepare input images and target domain labels.
                x_real = np.array(X).astype(np.float32)
                x_real = np.transpose(np.array(x_real), (0, 3, 1, 2))
                x_real = torch.tensor(x_real).half().to(device)
                x_real = normalize(x_real).to(device)

                c_trg_list = create_labels(label_org, c_dim)
                print('xreal', x_real.shape)
                (_, out_cls) = netD(x_real,0)
                print('out_cls', out_cls)
                _, predicted = torch.max(out_cls.data, 1)
                y_score = out_cls[:, 1].cpu()
                total += 1
                correct += (predicted.cpu() == label_org).sum().item()
                long_pred = torch.cat((long_pred, predicted.cpu()), dim=0)
                long_cls = torch.cat((long_cls, label_org), dim=0)
                long_score = torch.cat((long_score, y_score), dim=0)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append((netG(x_real, c_trg)))


                if label_org==0:
                    text='original healthy'
                    reconstruction = loss_metric(x_fake_list[1], x_fake_list[0])
                    total_rec += reconstruction
                    count_gesund+=1
                    diff = normalize(x_fake_list[1][0, 0, :, :]).cpu() - normalize(x_fake_list[0][0, 0, :, :]).cpu()
                    thresh = threshold_otsu(np.array(diff))
                    print(i, thresh)
                    varianz = diff.var()
                    total_var += varianz
                else:
                    text = 'original diseased'
                    diff = -normalize(x_fake_list[1][0, 0, :, :]).cpu() + normalize(x_fake_list[0][0, 0, :, :]).cpu()
                    thresh = threshold_otsu(np.array(abs(diff)))
                    thresholded_images = np.double(abs(diff) > 1 * abs(thresh))
                    print(i, thresh)


                if i%2==0 and label_org==1:             #plot output images
                    img = torch.zeros(7, 256, 256)
                    img[0, :, :] = x_fake_list[0][0, 0, :, :].cpu()
                    img[1, :, :] = x_fake_list[1][0, 0, :, :].cpu()
                    img[2, :, :] = x_fake_list[2][0, 0, :, :].cpu()
                    img[3, :, :] = diff
                    img[4, :, :] = torch.tensor(thresholded_images)

                    plt.figure(i)
                    ax = plt.subplot(2, 3, 1)
                    plt.imshow((normalize((x_fake_list[0][0, 0, :, :]))).cpu())
                    ax.axis('off')
                    ax.title.set_text(text)
                    ax = plt.subplot(2, 3, 2)
                    plt.imshow(normalize(x_fake_list[1][0, 0, :, :]).cpu())
                    ax.title.set_text('generated healthy')
                    ax.axis('off')
                    ax = plt.subplot(2, 3, 3)
                    plt.imshow(normalize(x_fake_list[2][0, 0, :, :]).cpu())
                    ax.axis('off')
                    ax.title.set_text('generated diseased')

                    ax = plt.subplot(2, 3, 4)
                    plt.imshow(diff)
                    ax.title.set_text('difference')
                    ax.axis('off')

            accuracy = 100 * correct / total
            auc = roc_auc_score(long_cls, long_score)
            (kappa, upper, lower) = kappa_score(long_pred, long_cls)
            avg_rec = total_rec / count_gesund
            avg_var = total_var / count_gesund
            
        print('AUROC', auc, 'accuracy', accuracy, 'kappa', kappa, 'mse reconstruction error', avg_rec, 'varianz gesund', avg_var)
        f = open('.descargan.txt', 'w')
        f.write('auroc '+str(auc)+'\n')
        f.write('accuracy '+str(accuracy)+'\n')
        f.write('MSE(a_h, r_h) ' + str(avg_rec) + '\n')
        f.write('kappa ' + str(kappa) + '\n')
        f.write('varianz reconstruction ' + str(avg_var) + '\n')







