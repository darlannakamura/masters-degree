import numpy as np
import torch
import torch.optim

# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from .models import *
from .utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

class DeepImagePrior:
    def run(self, iterations: int, image_noisy: np.ndarray):
        assert len(image_noisy.shape) == 3, "image_noisy should have 3 dimensions."
        assert image_noisy.shape[0] == 1, f"first shape should be equal to 1, but received {image_noisy.shape[0]}"

        INPUT = 'noise' # 'meshgrid'
        pad = 'reflection'
        OPT_OVER = 'net' # 'net,input'

        reg_noise_std = 1./30. # set to 1./20. for sigma=50
        LR = 0.01

        OPTIMIZER='adam' # 'LBFGS'
        show_every = 100
        exp_weight=0.99

        img_noisy_np = image_noisy
        img_noisy_pil = np_to_pil(img_noisy_np)
        
        num_iter = iterations
        input_depth = 1 
        
        net = get_net(input_depth, 'skip', pad,
                        skip_n33d=128, 
                        skip_n33u=128, 
                        skip_n11=4, 
                        num_scales=5,
                        upsample_mode='bilinear').type(dtype)

        net_input = get_noise(input_depth, INPUT, (img_noisy_pil.size[1], img_noisy_pil.size[0])).type(dtype).detach()

        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 

        # Loss
        mse = torch.nn.MSELoss().type(dtype)

        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        self.out_avg = None
        self.last_net = None
        self.psrn_noisy_last = 0
        self.net_input = net_input
        self.i = 0
        
        def closure():
            if reg_noise_std > 0: #reg_noise_std == 1./30
                self.net_input = net_input_saved + (noise.normal_() * reg_noise_std)
                #Ou seja, vamos a cada iteração tentar deixar o ruído cada vez mais parecido com o
                #que acreditamos ser o ruído real.

            out = net(self.net_input)
            
            # Smoothing
            if self.out_avg is None:
                self.out_avg = out.detach()
            else:
                #vamos a cada iteração levar em consideração 99% do anterior e 
                #1% da imagem de saída atual
                self.out_avg = self.out_avg * exp_weight + out.detach() * (1 - exp_weight)
                
            total_loss = mse(out, img_noisy_torch)
            total_loss.backward()

            psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
            
            # Note that we do not have GT for the "snail" example
            # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
            print ('Iteration %05d    Loss %f   PSNR_noisy: %f    ' % (self.i, total_loss.item(), psrn_noisy), '\r', end='')
            
            # Backtracking
            if self.i % show_every:
                if psrn_noisy - self.psrn_noisy_last < -5: 
                    print('Falling back to previous checkpoint.')

                    for new_param, net_param in zip(self.last_net, net.parameters()):
                        net_param.data.copy_(new_param.cuda())

                    return total_loss*0
                else:
                    self.last_net = [x.detach().cpu() for x in net.parameters()]
                    self.psrn_noisy_last = psrn_noisy
                    
            self.i += 1

            return total_loss

        p = get_params(OPT_OVER, net, self.net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)

        out_np_avg = torch_to_np(self.out_avg)
        out_np = torch_to_np(net(self.net_input))

        return out_np, out_np_avg
