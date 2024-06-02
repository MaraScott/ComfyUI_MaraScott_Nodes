#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###

from nodes import KSampler, CLIPTextEncode, VAEEncodeTiled, VAEDecodeTiled, ImageScale, SetLatentNoiseMask, ImageScaleBy

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