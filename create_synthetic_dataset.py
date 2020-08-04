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
from utils.tools import normalize, visualize, GaussianSmoothing
import os
import math
from random import  uniform
from utils.Functions import GaussianSmoothing,  save_tensor_image_as_png,save_tensor_image_as_png_gray
from matplotlib import pyplot as plt
from math import pi
import numpy as np
from shapely.geometry.polygon import LinearRing
os.system('mkdir warp_set')
os.system('mkdir warp_set/diseased')
os.system('mkdir warp_set/healthy')


smoothing = GaussianSmoothing(1, 3, 2, dim=2)
input=torch.ones(1,2,256,256)


count=0
x1 = np.linspace(0, 256, 256, endpoint=False)
x2 = np.linspace(0, 256, 256, endpoint=False)
for num in range(4000):
  if count<20:            #create a dataset of 2000 images
    tensor=torch.zeros(2000,2000)
    tensor2 = torch.zeros(2000, 2000)
    K = torch.zeros((2, 256, 256))
    f1 = uniform(0.2, 0.35)
    f2 = uniform(0.35, 0.5)
    h1= uniform(0.2, 0.4); h2= uniform(0.2, 0.4)

    #Centers of ellipses:
    u = np.array(torch.randint(50, 200, (1,)));
    v = np.array(torch.randint(50, 200, (1,)))  # Center of ellipse 1
    u2 = np.array(torch.randint(50, 200, (1,)));
    v2 = np.array(torch.randint(50, 200, (1,)))  # Center of ellipse2
    dist=np.sqrt((u-u2)**2+(v-v2)**2)

#-----------------------------------------------------------------------------------------------
# add Background
#-----------------------------------------------------------------------------------------------
    if dist>120:   #the ellipses should not be too close to each other, such that they do not intersect

        theta = np.linspace(0, 2 * np.pi, 10000)
        for r in range(500):
            cord1 = r * np.cos(theta)+u        #cord1 and cord2 are the coordinates on a circle
            cord2 = r * np.sin(theta)+v
            tensor[cord2,cord1] =math.sin(r*f1)*h1    #background in concentric waves c1

        tensor=tensor[:256,:256]


        for r in range(500):
            cord3 = r * np.cos(theta) + u2         #cord3 and cord4 are the coordinates on a circle
            cord4 = r * np.sin(theta) + v2
            tensor2[cord3, cord4] = math.sin(r *f2) * h2  #background in concentric waves c2

        tensor2=tensor2[:256,:256]

        tensor3=tensor+tensor2                  #for the background, the concentric waves c1 and c2 are added
        tensor4=tensor3.clone()


#  -----------------------------------------------------------------------------------------------------------
        #add ellipse 1
#  -----------------------------------------------------------------------------------------------------------

        angle=uniform(0, 2*np.pi)  #random rotation angle
        R  =torch.tensor( [[np.cos(angle), -np.sin(angle)],     #rotation matrix
        [np.sin(angle),  np.cos(angle)]])

        d1 =int( np.round(np.random.normal(5, 0.7, 1))) #contour thickness g for ellipse 1

        mua, sigmaa =40, 1; mub, sigmab =20, 1; # mean and standard deviation of axes of e1
        a = np.random.normal(mua, sigmaa, 1)    #first axis of e1
        b = np.random.normal(mub, sigmab, 1)    #second axis of e1

        t = np.linspace(0, 2*pi, 1000)

        for i in range(d1):

            x=(a-i)*np.cos(t)
            y=(b-i)*np.sin(t)
            C=torch.tensor([x,y])
            rCoords = torch.mm(R,C)
            xr = np.array(rCoords[0,:])
            yr = np.array(rCoords[1,:])

            e1=np.round(xr+u); e2=np.round(yr+v)
            if e1.max()<256 and  e2.max()<256:
                tensor3[e1, e2] = 1


#--------------------------------------------------------------------------------------------------
        #add ellipse 2
# --------------------------------------------------------------------------------------------------

        mua2, sigmaa2 =70, 1; mub2, sigmab2 =35,1; # mean and standard deviation of e2
        a2 = np.random.normal(mua2, sigmaa2, 1)    #first axis of e2
        b2 = np.random.normal(mub2, sigmab2, 1)    #first axis of e2
        angle2=uniform(0, 2*np.pi)               #random rotation angle
        R  =torch.tensor( [[np.cos(angle2), -np.sin(angle2)],   #Rotation matrix
        [np.sin(angle2),  np.cos(angle2)]])
        t = np.linspace(0, 2*pi, 1000)

        d2 =int( np.round(np.random.normal(5, 0.7, 1))) # contour thickness of e2

        for i in range(d2):

            x2=(a2-i)*np.cos(t)
            y2=(b2-i)*np.sin(t)
            C2=torch.tensor([x2,y2])
            rCoords = torch.mm(R,C2)
            xr2 = np.array(rCoords[0,:])
            yr2 = np.array(rCoords[1,:])


            e12=np.round(xr2+u2); e22=np.round(yr2+v2)
            if  e12.max()<256 and e22.max()<256:
                tensor3[e12,e22]=1
                tensor4[e12, e22] = 1

#-----------------------------------------------------------------------------------------------
        #Check whether ellipses intersect
# -----------------------------------------------------------------------------------------------

        def ellipse_polyline(ellipses, n=100):
            t = np.linspace(0, 2*np.pi, n, endpoint=False)
            st = np.sin(t)
            ct = np.cos(t)
            result = []
            for x0, y0, a, b, angle in ellipses:
                angle = np.deg2rad(angle)
                sa = np.sin(angle)
                ca = np.cos(angle)
                p = np.empty((n, 2))
                p[:, 0] = x0 + a * ca * ct - b * sa * st
                p[:, 1] = y0 + a * sa * ct + b * ca * st
                result.append(p)
            return result

        def intersections(a, b):
            ea = LinearRing(a)
            eb = LinearRing(b)
            mp = ea.intersection(eb)

            x = [p.x for p in mp]
            y = [p.y for p in mp]
            return x, y

        ellipses = [(u, v, a, b, angle*180/np.pi), (u2, v2, a2, b2, angle2*180/np.pi)]
        a3, b3 = ellipse_polyline(ellipses)
        x3, y3 = intersections(a3, b3)


        if len(x3)==0 and e2.max()<256 and e1.max()<256 and e12.max()<256 and e22.max()<256: #check whether the ellipses do not intersect and are inside the image

            T=smoothing(tensor3[None,None,:,:])

            #diff=T-smoothing(tensor3[None, None, :, :])

            K[0, :, :] = T[0,0,:,:]
            K2=K[None,...]
            X=K2[:,:1,:,:]
            count +=1
            print(count, 'count')
            plt.figure(1)
            plt.plot(T[0,0,...])
            plt.axis('off')

# -----------------------------------------------------------------------------------------
            # Apply Deformation Field to generate diseased images
 # -----------------------------------------------------------------------------------------

            def logpdf(x, mu, sigma):
                return -0.5 * (x - mu) * (x - mu) / (sigma * sigma) - 0.5 * torch.log(2 * math.pi * sigma * sigma)
            def gaus2d(x=0, y=0, mx=0.4, my=0.4, sx=0.19, sy=0.19):
                 return 1. / (2. * np.pi * sx * sy) * torch.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

            mu=torch.tensor(0)
            sigma=torch.tensor(0.1)

            # create meshgrid
            n =256

            x = torch.tensor(np.linspace(-1, 1,n)).float()
            y = torch.tensor(np.linspace(-1, 1,n)).float()
            x, y = torch.tensor(np.meshgrid(x, y))
            grid=torch.cat((x.unsqueeze(-1),y.unsqueeze(-1)), dim=2).unsqueeze(0)
            x.requires_grad = True
            y.requires_grad = True
            mv=torch.tensor(2*(u/256)-1).float()
            mu=torch.tensor(2*(v/256)-1).float()

            z=gaus2d(x,y, mx=mu, my=mv)


            # this line will compute the gradients
            torch.autograd.backward([z], [torch.ones(x.size()), torch.ones(y.size())])


            plt.figure(4)               #Quiver plot of the deformation field
            plt.quiver(x.detach(), y.detach(), -x.grad, -y.grad, z.detach(), alpha=.9)
            plt.axis('off')
            plt.show()

            g1=(-x.grad)
            g2=(-y.grad)
            disp2=torch.cat((g1[None,...,None], g2[None,...,None]), 3).float()
            disp2[disp2 != disp2] = 0
            print('disp2')
            vgrid = grid + disp2 / disp2.max() * 0.1

            output = torch.nn.functional.grid_sample(X, vgrid)

            Out=torch.tensor(output[0,0,:,:].detach().numpy())

            diff = -X[0,0,:,:]+ Out


            X = normalize(X)
            plt.figure(1)
            plt.subplot(1, 3, 1)
            plt.imshow(X[0, 0, :, :])
            plt.title('input healthy')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            output = normalize(Out)
            plt.subplot(1, 3, 2)
            plt.imshow(Out)
            plt.title('output diseased')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow((-diff), cmap='viridis')
            plt.title('Differenz')
            plt.axis('off')
            plt.show()
            print('done')
#------------------------------------------------------------------------------------------------
#Save generated images
#------------------------------------------------------------------------------------------------
            if count<2000:        #save 2000 images of healthy subjects
                K[0, :, :] = X[0, 0, :, :]
                K[1, :, :] = diff           #Ground Truth difference
                np.save( os.path.join('./warp_set/healthy', str(num)),K)

            else:                #save 2000 images of diseased subjects
                K[0, :, :] = Out
                K[1, :, :] = diff           #Ground Truth difference
                np.save( os.path.join('./warp_set/diseased', str(num)),K)
