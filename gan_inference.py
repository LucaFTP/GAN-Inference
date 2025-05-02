import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import  numpy as np
np.random.seed(2312)
import skimage as ski
from scipy.ndimage import rotate
from matplotlib import pyplot as plt
plt.style.use('dark_background')

from functions_for_inference import *

# TODO: initiate the training of the GAN using a configuration file (same as
# what has been done for the diffusion model. In this way, it is possible to 
# easily access the model configuration, as well as other parameters (for example the 
# mult_factor for the dynamic range optimization).

# Theese next parameter should eventually be read
# from the Gan configuration file for the trainig
END_SIZE = 128; latent_dim = 6; redshift = 0.03
# # # # # #

xgan = build_gan(END_SIZE=END_SIZE, noise_dim=latent_dim, ckp_path="pgan_0260.weights.h5")

### ----------------------------------------- ###
#|                                             |#
#|          INITIALIZATION OF VARIABLES        |#
#|                                             |#
### ----------------------------------------- ###

# Costruzione del target
sample_id = 'tSZ_sim=D8_snap=088_mass=14.49_reds=0.03_ax=z_rot=0.npy'
sample_img = np.load(sample_id)
target = ski.transform.resize(sample_img, (END_SIZE, END_SIZE))
target = rotate(target, angle=30, reshape=False)
# target = 10**target     # TNG maps are in log10 scale

sigma_r = 1e-3; sigma_i = 1e-4
real_target, imag_target = Fourier_transform(target)
real_target += np.random.normal(0, sigma_r, real_target.shape)
imag_target += np.random.normal(0, sigma_i, imag_target.shape)

mask = np.ones(shape=(END_SIZE, END_SIZE//2 + 1))  # modifica_linea_obliqua(matrix_size=END_SIZE)[:, (END_SIZE//2 - 1):]
bool_mask = np.array(mask, dtype=bool) # np.load("ALMA_spw_mask.npy")
masked_rx = real_target[bool_mask]; masked_ix = imag_target[bool_mask]

### ----------------------------------------- ###
#|                                             |#
#|                 LIKELIHOOD                  |#
#|                                             |#
### ----------------------------------------- ###

from scipy.stats import norm
from nautilus import Prior, Sampler

def modello(theta, gan_model):
    latent, mass = theta[:6], theta[6]
    sample = generate_image(mass=mass, model=gan_model, latent_vector=latent)
    sample_map = new_rescaling(gan_map=sample, mass=mass)
    model_real, model_imag = Fourier_transform(sample_map)
    return model_real, model_imag

def modello_no_mass(theta, gan_model):
    latent, p_factor = theta[:6], theta[6]
    sample = generate_image(mass=14.5, model=gan_model, latent_vector=latent)
    sample *= p_factor
    model_real, model_imag = Fourier_transform(sample)
    return model_real, model_imag

## Implementation for nautilus
prior = Prior()
prior.add_parameter(r'$\theta_1$', dist=(-3.2, 3.2))
prior.add_parameter(r'$\theta_2$', dist=(-3.2, 3.2))
prior.add_parameter(r'$\theta_3$', dist=(-3.2, 3.2))
prior.add_parameter(r'$\theta_4$', dist=(-3.2, 3.2))
prior.add_parameter(r'$\theta_5$', dist=(-3.2, 3.2))
prior.add_parameter(r'$\theta_6$', dist=(-3.2, 3.2))
prior.add_parameter('logM500', dist=(13.5, 15.5))

def nautilus_likelihood(param_dict):
    latent = [param_dict[r'$\theta_1$'], param_dict[r'$\theta_2$'], param_dict[r'$\theta_3$'],
              param_dict[r'$\theta_4$'], param_dict[r'$\theta_5$'], param_dict[r'$\theta_6$']]
    mass = param_dict['logM500']

    real_from_model, imag_from_model = modello(latent + [mass], xgan)
    real_from_model = real_from_model[bool_mask]; imag_from_model = imag_from_model[bool_mask]
    logL_real = np.nansum(norm.logpdf(masked_rx, loc=real_from_model, scale=sigma_r))
    logL_imag = np.nansum(norm.logpdf(masked_ix, loc=imag_from_model, scale=sigma_i))

    return logL_real + logL_imag

### ----------------------------------------- ###
#|                                             |#
#|                   MAIN                      |#
#|                                             |#
### ----------------------------------------- ###

import corner
import argparse
from mpi4py.futures import MPIPoolExecutor

if __name__ == "__main__":

    # Parser creation
    parser = argparse.ArgumentParser(
        description="Imposta i parametri per l'esecuzione del programma."
    )
    # Parser parameter definition
    parser.add_argument("--n_proc", type=int, default=4,
                        help="Number of cores for parallel computation (default: 4).")
    parser.add_argument("-t", "--timeout", type=int, default=60,
                        help="Timeout time in minutes (default: 60).")
    parser.add_argument("-plt", "--plot_corner", type=bool, default=True,
                        help="Option for plotting the corner_plot with corner library (default: True).")
    parser.add_argument("-b", "--burnin", type=int, default=50000,
                        help="Burnin value (it will be taken negative, so represents the last n steps).")
    
    # Arguments parsing
    args = parser.parse_args()
    
    ckpt_file = 'D8_noise=r3_i4_30degrees_newckpt_nosparse.h5'

    n_proc = args.n_proc
    sampler = Sampler(prior, nautilus_likelihood, pool=MPIPoolExecutor(max_workers=n_proc), filepath=ckpt_file)
    sampler.run(verbose=True, timeout=60*args.timeout)

    if args.plot_corner:
        points, log_w, log_l = sampler.posterior()
        
        names = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$',
          r'$\theta_5$', r'$\theta_6$', '$log(M_{500}/M_{\odot})$']
        
        burnin = - args.burnin
        samps = points[burnin:, :] 

        figure = corner.corner(samps,
                       weights=np.exp(log_w[burnin:]),
                       labels=names,
                       show_titles=True,
                       quantiles=[0.16, 0.5, 0.84],
                       fill_contours=True
                       )
        plt.savefig(f'images/corner_{ckpt_file.split('.')[0]}.png')
        plt.close()
