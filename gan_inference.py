# TODO: initiate the training of the GAN using a configuration file (same as
# what has been done for the diffusion model. In this way, it is possible to 
# easily access the model configuration, as well as other parameters (for example the 
# mult_factor for the dynamic range optimization).

import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import  numpy as np
np.random.seed(2312)
from scipy import fftpack
from matplotlib import pyplot as plt
plt.style.use('dark_background')

import tensorflow as tf

### ----------------------------------------- ###
#|                                             |#
#|              UTILS FUNCTIONS                |#
#|                                             |#
### ----------------------------------------- ###

def inv_dynamic_range(synth_img, eps=1e-6, mult_factor=1):
    image = np.asarray(synth_img)/mult_factor
    a = - eps**(1 - image)
    b = eps**image - 1
    return a*b

def _create_circular_mask(h, w, radius=None, center=None):

        if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

### ----------------------------------------- ###
#|                                             |#
#|               SZ_map CLASS                  |#
#|                                             |#
### ----------------------------------------- ###

import scipy.integrate
from szint.szint import get_ysz_line
from photutils.profiles import RadialProfile
from astropy.cosmology import Planck18 as cosmo

class SZ_map(np.ndarray):

    def __new__(cls, input_image, redshift, m500, model='a10_up'):
        obj = np.asanyarray(input_image).view(cls)
        obj.redshift = redshift
        obj.m500 = 10**m500
        obj.center = (obj.shape[0] // 2, obj.shape[1] // 2)
        obj.model = model
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.redshift = getattr(obj, 'redshift', None)
        self.m500 = getattr(obj, 'm500', None)
        self.center = getattr(obj, 'center', None)
        self.model = getattr(obj, 'model', None)

    def evaluate_at_r(self, r, cosmo=cosmo):
        """Calcola il valore dell'integrale lungo la linea di vista a distanza r."""
        return get_ysz_line(r, self.m500, self.redshift, model=self.model, cosmo=cosmo)[1]

    def generate_radial_profiles(self, edge_step=4):
        """Genera profili radiali basati su Photutils."""
        edge_radii = np.arange(0, self.shape[0] // 2, edge_step)
        rp = RadialProfile(self, self.center, edge_radii, mask=None)  # Usa self direttamente
        return rp.radius, rp.profile

    def window_integral(self, r, type: str, radius_unit='r500'):
        """
        Restituisce il segnale integrato all'interno di una regione centrata.
        """
        radius_conversion = {
            'pixel': (r, (r * 5) / self.shape[0]),
            'r500': ((r * self.shape[0]) / 5, r)
        }

        if radius_unit not in radius_conversion:
            raise ValueError(f"radius_unit '{radius_unit}' non supportato. Usa 'pixel' o 'r500'.")

        r_d, r_a = radius_conversion[radius_unit]

        if type == 'discrete':
            mask = _create_circular_mask(h=self.shape[0], w=self.shape[1], radius=r_d)
            masked_image = self.copy()  # Usa self invece di self.input_image
            masked_image[~mask] = 0
            w_int = np.sum(masked_image) * (100 / (self.shape[0] * self.shape[1]))

        elif type == 'analytic':
            w_int = 2 * np.pi * scipy.integrate.quad(lambda x: self.evaluate_at_r(x) * x, 0, r_a)[0]

        else:
            raise ValueError(f"type '{type}' non supportato. Usa 'discrete' o 'analytic'.")

        return w_int

    def plot(self, save=False, filename=None):
        """Visualizza o salva l'immagine della mappa SZ."""
        plt.imshow(self, cmap='inferno')

        if save:
            if filename is None:
                filename = f'SZ_map_z_{self.redshift:.2f}_m500_{self.m500:.2e}.png'
            plt.imsave(filename, self, cmap='inferno')
 
def rescale(
          sz_map:SZ_map, r=0.75, model=None
          ) -> SZ_map:
    '''Apply correction to the pixel values according to a model. Default Arnaud profile.e'''
    if model==None or model=='a10_up':
        ref_value = sz_map.window_integral(r, 'analytic')
    elif model=='g16':
        ref_value = g16_Y_cyl(sz_map.m500, sz_map.redshift)
    elif model=='dianoga_fit':
        # TODO: this next section needs to become more flexible, in order to 
        # accomodate for different redshift values
        Dian_sim_fit = lambda m: -3.708 + 0.796*(np.log10(m) - 14.85) # Fit results within 5*r_500 for z=0.03
        ref_value = 10**Dian_sim_fit(sz_map.m500)
    synth_value = sz_map.window_integral(r, 'discrete')
    sz_map = sz_map * ref_value / synth_value
    return sz_map
    
def g16_Y_cyl(m500, z, A = -4.697, B = 1.65, C = 0.45):
    '''Gupta+16 fit results'''
    a = 10**A
    b = (m500 / (3 * 10**14))**B
    c = (cosmo.efunc(z) / cosmo.efunc(0.6))**C
    return a*b*c

### ----------------------------------------- ###
#|                                             |#
#|          GAN MODEL INITIALIZATION           |#
#|                                             |#
### ----------------------------------------- ###

from model import PGAN

def build_gan(
        END_SIZE:int, noise_dim:int, ckp_path:str
        ) -> PGAN:
    
    xgan = PGAN(latent_dim = noise_dim)

    for n_depth in range(1, int(np.log2(END_SIZE/2))):
        xgan.n_depth = n_depth
        xgan.fade_in_generator()
        xgan.fade_in_discriminator()
        xgan.fade_in_regressor()
        
        xgan.stabilize_discriminator()
        xgan.stabilize_generator()
        xgan.stabilize_regressor()

    xgan.build(input_shape=(END_SIZE, END_SIZE))
    xgan.load_weights(ckp_path)    
    
    return xgan

### ----------------------------------------- ###
#|                                             |#
#|                LATENT SAMPLING              |#
#|                                             |#
### ----------------------------------------- ###

def generate_image(
        mass:float, model:PGAN, latent_vector:np.ndarray
        ) -> np.ndarray:
    
    mass_tensor = tf.convert_to_tensor(np.expand_dims(mass, 0), dtype=tf.float32)
    random_latent_vectors = tf.convert_to_tensor(np.expand_dims(latent_vector, 0), dtype=tf.float32)
    image = model.generator([random_latent_vectors, mass_tensor])
    
    return np.squeeze(inv_dynamic_range(image, mult_factor=2.5))

def Fourier_transform(
        image:np.ndarray, out:str = 'mod_phi'
        ) -> tuple[np.ndarray, np.ndarray]:
    '''
    This function simply applies fftpack.fft2 and returns module and phase
    (or real and imaginary part if specified).
    image: np.ndarray over which applying the fft
    out: str that the definies the type of output (default mod and phase, available also 're_im')
    '''
    image_fft = np.fft.fftshift(fftpack.fft2(image))
    if out == 'mod_phi': return np.absolute(image_fft), np.angle(image_fft)
    elif out == 're_im': return np.real(image_fft), np.imag(image_fft)
    else: sys.exit("Unrecongnized Fourier type output")

### ----------------------------------------- ###
#|                                             |#
#|          INITIALIZATION OF VARIABLES        |#
#|                                             |#
### ----------------------------------------- ###

# Theese next parameter should eventually be read
# from the Gan configuration file for the trainig
END_SIZE, latent_dim, redshift = 128, 6, 0.03
# # # # # #


# Costruzione del target
sample_id = 'tSZ_sim=D8_snap=088_mass=14.49_reds=0.03_ax=z_rot=0.npy'
sample_img = np.load(sample_id)
resize_factor = sample_img.shape[0] // END_SIZE
target = sample_img[::resize_factor, ::resize_factor]

xgan = build_gan(END_SIZE=END_SIZE, noise_dim=latent_dim, ckp_path="pgan_0990.weights.h5")
'''
# Costruzione del target
fixed_latent = np.asarray([2, 1.2, 0, 0, 0, 0]); fixed_mass = 14.5
target = generate_image(fixed_mass, xgan, fixed_latent)
target_map = rescale(SZ_map(target, redshift=redshift, m500=fixed_mass), model='dianoga_fit')

'''
# Test in image space
sigma = 5e-6
noisy_obs = target + np.random.normal(0, sigma, target.shape)

### ----------------------------------------- ###
#|                                             |#
#|                 LIKELIHOOD                  |#
#|                                             |#
### ----------------------------------------- ###

from nautilus import Prior, Sampler
from getdist import plots, MCSamples
from scipy.stats import norm, uniform

def modello(theta, gan_model):
    # TODO: redshift value inserted by hand, not very flexible
    latent, mass = theta[:6], theta[6]
    # mass = theta
    # latent = np.random.normal(0, 1, 6)
    sample = generate_image(mass=mass, model=gan_model, latent_vector=latent)
    sample_map = rescale(SZ_map(input_image=sample, redshift=0.03, m500=mass), model='dianoga_fit')
    # sample_module, sample_phase = Fourier_transform(sample_map, out='re_im')
    sample_rfft = np.fft.rfft2(sample_map)
    return sample_rfft.real, sample_rfft.imag# sample_module, sample_phase

def modello_image_space(theta, gan_model):
    latent, mass = theta[:6], theta[6]
    sample = generate_image(mass=mass, model=gan_model, latent_vector=latent)
    sample_map = rescale(SZ_map(input_image=sample, redshift=0.03, m500=mass), model='dianoga_fit')

    return sample_map

## Implementation for nautilus
prior = Prior()
prior.add_parameter(r'$\theta_1$', dist=(-3, 3))
prior.add_parameter(r'$\theta_2$', dist=(-3, 3))
prior.add_parameter(r'$\theta_3$', dist=(-3, 3))
prior.add_parameter(r'$\theta_4$', dist=(-3, 3))
prior.add_parameter(r'$\theta_5$', dist=(-3, 3))
prior.add_parameter(r'$\theta_6$', dist=(-3, 3))
prior.add_parameter('logM500', dist=(13.5, 15.5))

def nautilus_likelihood(param_dict):
    latent = [param_dict[r'$\theta_1$'], param_dict[r'$\theta_2$'], param_dict[r'$\theta_3$'],
              param_dict[r'$\theta_4$'], param_dict[r'$\theta_5$'], param_dict[r'$\theta_6$']]
    mass = param_dict['logM500']

    image_from_model = modello_image_space(latent + [mass], xgan)
    logL_image_space = np.nansum(norm.logpdf(noisy_obs, loc=image_from_model, scale=sigma))

    return logL_image_space

def log_prior(theta):
    params, mass = theta[:6], theta[6]
    # mass = theta

    # Gaussian prior on latent vector components and module noise
    # lp = np.sum(norm.logpdf(params, loc=0, scale=1))
    if np.all((-3.2 <= params) & (params <= 3.2)):
        lp = 0
    else:
        return -np.inf

    # Uniform prior over mass and delta_phi
    if 13.8 <= mass <= 15:
        lp += 0
    else:
        return -np.inf

    return lp

"""
According to emcee documentation, https://emcee.readthedocs.io/en/stable/tutorials/parallel/
it is better to define the posterior variables as global variables instead of having to call
them at every iteration in the multiprocessing pool.
"""

def log_likelihood(theta, gan_model):
    
    image_from_model = modello_image_space(theta, gan_model)
    logL = np.nansum(norm.logpdf(noisy_obs, loc=image_from_model, scale=sigma))

    return logL

def log_posterior(theta):

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, xgan)

### ----------------------------------------- ###
#|                                             |#
#|                   MAIN                      |#
#|                                             |#
### ----------------------------------------- ###

import emcee
from mpi4py.futures import MPIPoolExecutor

if __name__ == "__main__":

    '''
    filename = 'storage.h5'
    backend = emcee.backends.HDFBackend(filename=filename)
    ndim = 7; n_walkers = 32; n_steps = 5000

    theta0 = np.asarray([2, 1.2, 0, 0, 0, 0]) + np.random.normal(loc=0, scale=1, size=(n_walkers, 6))
    mass0 = 14.2 + np.random.normal(loc=0, scale=0.5, size=(n_walkers, 1))
    pos = np.hstack((theta0, mass0))
    
    n_proc = 4
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, pool=MPIPoolExecutor(max_workers=n_proc), backend=backend)
    sampler.run_mcmc(pos, n_steps, progress=True)

    '''
    ckpt_file = 'naut_ckpt_sim_image.h5'

    n_proc = 4
    sampler = Sampler(prior, nautilus_likelihood, pool=MPIPoolExecutor(max_workers=n_proc), filepath=ckpt_file)
    sampler.run(verbose=True, timeout=4.7e3)

    '''
    points, log_w, log_l = sampler.posterior()
    
    names = ['theta_1', 'theta_2', 'theta_3', 'theta_4',
         'theta_5', 'theta_6', 'logM500']
    
    burnin = int(np.round(len(points[:, 0]) / 3))
    samps = points[burnin:, :]
    samples = MCSamples(samples=samps, names = names)
    g = plots.get_subplot_plotter()
    g.triangle_plot([samples], filled=True)
    g.export('output_file_5e3.pdf')
    '''