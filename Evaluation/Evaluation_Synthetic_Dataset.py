import torch
import numpy as np
import torchvision
from visdom import Visdom
import matplotlib.pyplot as plt
import os
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
viz=Visdom(port=8850)
from utils.tools import  npy_loader,  normalize, visualize, eval_binary_classifier, kappa_score
import torch.nn as nn
from utils.Functions import classification_loss,  label2onehot, create_labels
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
        netG=Generator().to(device)
        netD=Discriminator().to(device)
        path = os.path.join(self.dataset_path, 'test')
        print(path, 'PATH')
        os.getcwd()

        path2 = self.choose_net

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


        g=0

        g_optimizer = torch.optim.Adam(netG.parameters(), 0.0001, [beta1, beta2])
        d_optimizer = torch.optim.Adam(netD.parameters(), 0.0001, [beta1, beta2])
        netG, g_optimizer = amp.initialize(netG, g_optimizer, opt_level='O1')
        netD, d_optimizer = amp.initialize(netD, d_optimizer, opt_level='O1')
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        netG = torch.nn.DataParallel(netG, device_ids=[id], output_device=id)
        netD = torch.nn.DataParallel(netD, device_ids=[id], output_device=id)


        try_loading_file=True
        if try_loading_file:

            try:

                netG.load_state_dict(torch.load(os.path.join(path2, 'netG_synthetic.pt'), map_location={'cuda:0': 'cpu'}))  # does NOT load optimizer state etc.
                netD.load_state_dict(torch.load(os.path.join(path2, 'netD_synthetic.pt'), map_location={'cuda:0': 'cpu'}))

                print("loaded model from file")
            except:
                print("loading model from file failed; created new model")

        c_dim=2
        #



# =================================================================================== #
#                                 5. Testing                                          #
# =================================================================================== #


        """Translate images using StarGAN trained on a single dataset."""

        loss_metric = nn.MSELoss()
        with torch.no_grad():
             sum_dice = 0;
             count = 0
             correct = 0
             total = 0; total_auc=0
             netD = netD.eval(); netG = netG.train()
             total_rec = 0;total_diff = 0; total_var=0; total_var2=0;sum_ssim=0;threshtot=0
             count_krank = 0;count_gesund = 0

             for i, (X3, c_org) in enumerate(test_loader):

                 GT = -X3[:, 1, :, :]
                 thresh = threshold_otsu(np.array(abs(GT[0,:,:])))
                 threshtot+=thresh
                 x_real = torch.tensor(X3[:, :1, :, :]).to(device)
                 noise = torch.rand(x_real.shape).to(device)*0
                 x_real = (x_real + 0.05 * noise).half()
                 (_, out_cls) = netD(x_real,g)
                 _, predicted = torch.max(out_cls.data, 1)
                 correct += (predicted.cpu() == c_org).sum().item()
                 total+=1
                 avg_thresh=0.44195

                 c_trg_list = create_labels(c_org, c_dim)

        #         # Translate images.
                 x_fake_list = [x_real]
                 for c_trg in c_trg_list:
                     x_fake_list.append((netG(x_real, c_trg)))
        #

                 if c_org == 0:

                     diff = normalize(x_fake_list[0][0, 0, :, :]).cpu() - normalize (x_fake_list[1][0, 0, :,:]).cpu()  # check whether 2 ist krank (ich glaub schon). Dann nimm immer die Differenz zwischen krank und Original

                     thresholded_images = np.double(abs(diff) > avg_thresh)*1
                     GTthresh = np.double(abs(GT )> avg_thresh) * 1
                     reconstruction = loss_metric(normalize(x_fake_list[1]), normalize(x_fake_list[0]))
                     varianz=diff.var()
                     total_rec += reconstruction
                     total_var+=varianz
                     count_gesund += 1

                 else:
                     diff = normalize(x_fake_list[0][0, 0, :, :]).cpu() - normalize(x_fake_list[1][0, 0, :, :]).cpu()  # check whether 2 ist krank (ich glaub schon). Dann nimm immer die Differenz zwischen krank und Original

                     thresh = threshold_otsu(np.array(abs(diff)))
                     print(i, thresh, 2 * abs(thresh))
                     thresholded_images = np.double(abs(diff) > avg_thresh)*1#1 * abs(thresh)) * 1
                     GTthresh = np.double(abs(GT) > avg_thresh) * 1
                     region = loss_metric(diff, GT[0, :, :])
                     total_diff += region
                     varianz2 = (normalize(np.array(GT[0, :, :])) - normalize(np.array(diff))).var()
                     total_var2 += varianz2

                     count_krank += 1
                     (output_DSC, avg) = eval_binary_classifier(np.array(GTthresh[0, :, :]), thresholded_images)
                     sum_dice += output_DSC['DSC'];
                     count += 1

                     ssim_val = ssim(visualize(diff[None,None,...]), visualize(GT[None,...]), data_range=1, size_average=False)
                     sum_ssim+=ssim_val


                     pixel_wise_cls = np.array(torch.tensor(visualize(abs(diff))).view(1, -1))[0, :]
                     pixel_wise_gt = np.array(torch.tensor(GTthresh).view(1, -1))[0, :]

                     auc = roc_auc_score(pixel_wise_gt, pixel_wise_cls)
                     print('auc', i, auc)
                     total_auc += auc
        #
                 if i%10==0:

                     plt.figure(i)      #plot results
                     ax = plt.subplot(2, 4, 1)
                     plt.imshow((normalize((x_fake_list[0][0, 0, :, :]))).cpu())
                     ax.title.set_text('original')
                     ax = plt.subplot(2, 4, 2)
                     plt.imshow(normalize(x_fake_list[1][0, 0, :, :]).cpu())
                     ax.title.set_text('label 0')
                     ax.axis('off')
                     ax = plt.subplot(2, 4, 3)
                     plt.imshow(normalize(x_fake_list[2][0, 0, :, :]).cpu())
                     ax.axis('off')
                     ax.title.set_text('label 1')

                     ax = plt.subplot(2, 4, 5)
                     plt.imshow(thresholded_images)
                     ax.title.set_text('differenz thresholded')
                     ax = plt.subplot(2, 4, 6)
                     plt.imshow(GT[0, :, :])
            #
                     ax.title.set_text('Ground Truth')
                     ax = plt.subplot(2, 4, 4)
                     plt.imshow(diff)
                     ax.title.set_text('difference')
                     if c_org != 0:
                        plt.suptitle(auc)
                     ax = plt.subplot(2, 4, 8)
                     plt.imshow(GTthresh[0, :, :])
                     ax.title.set_text('GTthresh')

        accuracy= 100 * correct / total
        avg_diff=total_diff/count_krank
        avg_auc=total_auc/count_krank
        avg_rec=total_rec/count_gesund
        avg_var=total_var/count_gesund
        avg_var2=total_var2/count_krank
        avg_ssim=sum_ssim/count_krank
        avg_dice=sum_dice/count
        print('average mse reconstruction error', avg_rec,'average mse in segmentation', avg_diff, 'average Dice' ,avg_dice, 'classification accuracy', accuracy, 'AUROC', avg_auc, 'SSIM', avg_ssim,'varianz gesund', avg_var,'varianz krak', avg_var2 )
        f = open('./descargan.txt', 'w')
        f.write('auroc '+str(auc)+'\n')
        f.write('MSE(a_h, r_h) ' + str(avg_rec) + '\n')
        f.write('varianz reconstruction ' + str(avg_var) + '\n')
        f.write('varianz difference ' + str(avg_var2) + '\n')
        f.write('Dice ' + str(avg_dice) + '\n')
        f.write('classification accuracy ' + str(accuracy) + '\n')
        f.write('MSE(gt, d) ' + str(avg_diff) + '\n')
        f.write('SSIM ' + str(avg_ssim) + '\n')








