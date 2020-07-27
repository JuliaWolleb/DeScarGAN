import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sig = nn.Sigmoid()
device='cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)


class Identity(nn.Module):

    def forward(self, x):
        return x

ACTIVATION = nn.ReLU
c_dim=2

def crop_and_concat(upsampled, bypass,crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)


def conv2d_bn_block(in_channels, out_channels,kernel=3, momentum=0.01, activation=ACTIVATION):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,kernel , padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION):

    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):

    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )



def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )




class Generator(nn.Module):


        def __init__(self, n_channels=1, nf=64, batch_norm=True, activation=ACTIVATION):
            super(Generator, self).__init__()

            act = activation

            conv_block = conv2d_bn_block if batch_norm else conv2d_block

            max_pool = nn.MaxPool2d(2)
            act = activation
            self.label_k=torch.tensor([0,1]).half().cuda()

            self.down0 = nn.Sequential(
                conv_block(n_channels + c_dim, nf, activation=act),
                conv_block(nf, nf, activation=act)
            )
            self.down1 = nn.Sequential(
                max_pool,
                conv_block(nf, 2 * nf, activation=act),
                conv_block(2 * nf, 2 * nf, activation=act),
            )
            self.down2 = nn.Sequential(
                max_pool,
                conv_block(2 * nf, 4 * nf, activation=act),
                conv_block(4 * nf, 4 * nf, activation=act),


            )
            self.down3 = nn.Sequential(
                max_pool,
                conv_block(4 * nf, 8 * nf, activation=act),
                conv_block(8 * nf, 8 * nf, activation=act),
            )

            self.up3 = deconv2d_bn_block(8 * nf, 4 * nf, activation=act)

            self.conv5 = nn.Sequential(
                conv_block(8 * nf, 4 * nf, activation=act),  # 8
                conv_block(4 * nf, 4 * nf, activation=act),
            )
            self.up2 = deconv2d_bn_block(4 * nf, 2 * nf, activation=act)
            self.conv6 = nn.Sequential(
                conv_block(4* nf, 2 * nf, activation=act),
                conv_block(2 * nf, 2 * nf, activation=act),
            )


            self.up1 = deconv2d_bn_block( 2*nf, nf, activation=act)
   

            self.conv7_k = nn.Sequential(
                  conv_block( nf, nf, activation=act),
                  conv_block(nf, n_channels, activation=nn.Tanh),
              )
            
            self.conv7_g = nn.Sequential(
                  conv_block( nf, nf, activation=act),
                  conv_block(nf, n_channels, activation=nn.Tanh),
              )

        def forward(self, x, c):
            c1 = c.view(c.size(0), c.size(1), 1, 1)
            c1 = c1.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat([x, c1], dim=1)

            x0 = self.down0(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)


            xu3 = self.up3(x3)
            cat3 = crop_and_concat(xu3, x2)
            x5 = self.conv5(cat3)
            xu2 = self.up2(x5)
            cat2 = crop_and_concat(xu2, x1)

            x6 = self.conv6(cat2)
            xu1 = self.up1(x6)
            cat1 = crop_and_concat(xu1, x0)

            if self.label_k  in c:
                x7 = self.conv7_k(xu1)
            else:
                x7 = self.conv7_g(xu1)

            return x7


class Discriminator(nn.Module):

            def __init__(self, n_channels=1, init_filters=32, batch_norm=False):
                super(Discriminator, self).__init__()
                nf = init_filters
                self.label_k = torch.ones(1).long().cuda()

                conv_block = conv2d_bn_block if batch_norm else conv2d_block

                max_pool = nn.MaxPool2d
                self.encoder = nn.Sequential(
                    conv_block(n_channels, nf),
                    max_pool(2),
                    conv_block(nf, 2 * nf),

                    max_pool(2),
                    conv_block(2 * nf, 4 * nf),
                    conv_block(4 * nf, 4 * nf),
                    max_pool(2),
                    conv_block(4 * nf, 8 * nf),
                    conv_block(8 * nf, 8 * nf),
                    max_pool(2),
                    conv_block(8 * nf, 8 * nf),
                    conv_block(8 * nf, 8 * nf),
                    max_pool(2),
                    conv_block(8 * nf, 16 * nf),
                

                )
                kernel_size = int(240 / np.power(2, 5))
                self.conv_k = nn.Sequential( conv_block(16 * nf, 16 * nf),conv_block(16 * nf,16* nf),conv_block(16 * nf, 1, kernel=1, activation=Identity),)
                self.conv_g = nn.Sequential(conv_block(16 * nf, 16 * nf),conv_block(16 * nf,16* nf), conv_block(16 * nf, 1, kernel=1, activation=Identity), )
                self.conv2=nn.Sequential(conv_block(16 * nf, 16 * nf), conv_block(16 * nf,16 * nf), max_pool(2),)
         

                self.linearclass = nn.Sequential(

                    Flatten(),
                    nn.Linear(512 * 4 * 4, 64),
                     nn.ReLU(True),
                     nn.Dropout(p=0.1),
                     nn.Linear(64, 2),

                )

            def forward(self, x,label):
                   h = self.encoder(x)
                   if label == self.label_k:
                     out = self.conv_k(h)

                   else:
                     out = self.conv_g(h)
                   zwischen=self.conv2(h)

                   klasse=self.linearclass(zwischen)

                   return out, klasse.reshape(klasse.size(0), klasse.size(1))


