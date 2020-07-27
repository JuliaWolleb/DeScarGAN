import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
sig = nn.Sigmoid()
import math
device='cuda'
c_dim=2



def classification_loss(logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)



def gradient_penalty( y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).half().to(device)

    z = torch.autograd.grad(outputs=y,
                              inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True, allow_unused=True,
                               only_inputs=True)[0]
   
    z = z.view(z.size(0), -1)
    z_l2norm = torch.sqrt(torch.sum(z ** 2, dim=1))
 
    return torch.mean((z_l2norm - 1) ** 2)


def label2onehot( labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


def create_labels( c_org, c_dim=2):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.

    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def update_lr( g_lr, d_lr, g_optimizer, d_optimizer):
        """Decay learning rates of the generator and discriminator."""
        for param_group in g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = d_lr


class GaussianSmoothing(nn.Module):


            def __init__(self, channels, kernel_size, sigma, dim=2):
                super(GaussianSmoothing, self).__init__()
                if isinstance(kernel_size, numbers.Number):
                    kernel_size = [kernel_size] * dim
                if isinstance(sigma, numbers.Number):
                    sigma = [sigma] * dim

                # The gaussian kernel is the product of the
                # gaussian function of each dimension.
                kernel = 1
                meshgrids = torch.meshgrid(
                    [
                        torch.arange(size, dtype=torch.float32)
                        for size in kernel_size
                    ]
                )
                for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
                    mean = (size - 1) / 2
                    kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                              torch.exp(-((mgrid - mean) / std) ** 2 / 2)

                # Make sure sum of values in gaussian kernel equals 1.
                kernel = kernel / torch.sum(kernel)

                # Reshape to depthwise convolutional weight
                kernel = kernel.view(1, 1, *kernel.size())
                kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

                self.register_buffer('weight', kernel)
                self.groups = channels

                if dim == 1:
                    self.conv = F.conv1d
                elif dim == 2:
                    self.conv = F.conv2d
                elif dim == 3:
                    self.conv = F.conv3d
                else:
                    raise RuntimeError(
                        'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
                    )

            def forward(self, input):

                return self.conv(input, weight=self.weight, padding=1, groups=self.groups)

def scalar_2_vector(img_src, colormap):
    cm_hot = mpl.cm.get_cmap(colormap)
    img_src.thumbnail((1024,1024))
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    print(im.max(), im.min())
    im = Image.fromarray(im)
    return im

def save_tensor_image_as_png(image, filename, colormap='viridis'):
    im = Image.fromarray(image)
    im = scalar_2_vector(im, colormap)
    plt.figure(5)
    plt.imshow(im)
    im.save(filename)

def save_tensor_image_as_png_gray(image, filename, colormap='gray'):
        im = Image.fromarray(image)
        im = scalar_2_vector(im, colormap)
        plt.figure(5)
        plt.imshow(im)
        im.save(filename)

