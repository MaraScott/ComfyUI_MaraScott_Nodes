#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from nodes import KSampler, VAEEncode, VAEEncodeTiled, VAEDecode, VAEDecodeTiled, ImageScaleBy
from comfy_extras.nodes_custom_sampler import *


from ...utils.log import *

class MS_Sampler:
    
    @classmethod
    def refine(self, image, upscale_method, vae, tile_size, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise):
        
        inpainted = ImageScaleBy().upscale(image, upscale_method, 1.5)[0]
        latent = VAEEncodeTiled().encode(vae, inpainted, tile_size)[0]
            
        latent = KSampler().sample(
            model, 
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent,
            denoise
        )[0]

        inpainted = VAEDecodeTiled().decode(vae, latent, tile_size)[0]
        return ImageScaleBy().upscale(inpainted, upscale_method, (1/1.5))
    
    def refine_custom(self, image, tiled, upscale_method, vae, tile_size, model, noise_seed, cfg, sampler, sigmas, positive, negative):
        
        add_noise = True
        upscale = ImageScaleBy().upscale(image, upscale_method, 1.5)[0]

        if tiled == True:
            latent_image = VAEEncodeTiled().encode(vae, upscale, tile_size)[0]
        else:
            latent_image = VAEEncode().encode(vae, upscale, tile_size)[0]
            
        latent_image = SamplerCustom().sample(
            model, 
            add_noise, 
            noise_seed, 
            cfg, 
            positive, 
            negative, 
            sampler, 
            sigmas, 
            latent_image
        )[0]

        if tiled == True:
            output = VAEDecodeTiled().decode(vae, latent_image, tile_size)[0] # .unsqueeze(0)
        else:
            output = VAEDecode().decode(vae, latent_image, tile_size)[0]

        return ImageScaleBy().upscale(output, upscale_method, (1/1.5))    