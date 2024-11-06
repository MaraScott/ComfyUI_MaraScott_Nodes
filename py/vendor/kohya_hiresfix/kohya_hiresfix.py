# ref https://gist.github.com/laksjdjf/487a28ceda7f0853094933d2e138e3c6
import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed, timestep_embedding, th

def apply_control(h, control, name):
    if control is not None and name in control and len(control[name]) > 0:
        ctrl = control[name].pop()
        if ctrl is not None:
            ctrl = torch.nn.functional.interpolate(ctrl.float(), size=(h.shape[2], h.shape[3]), mode="bicubic", align_corners=False).to(h.dtype)
            h += ctrl
    return h

class Hires:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ds_depth_1": ("INT", {
                    "default": 3,
                    "min": -1,
                    "max": 12,
                    "step": 1,
                    "display": "number"
                }),
                "ds_depth_2": ("INT", {
                    "default": 3,
                    "min": -1,
                    "max": 12,
                    "step": 1,
                    "display": "number"
                }),
                "ds_timestep_1": ("INT", {
                    "default": 900,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "ds_timestep_2": ("INT", {
                    "default": 650,
                    "min": 0,
                    "max": 1000,
                    "step": 0.1,
                }),
                "resize_scale_1": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 16.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "resize_scale_2": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 16.0,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply"
    CATEGORY = "loaders"
    
    def hires_resize(self, h, timestep, depth):
        dtype = h.dtype
        if timestep > self.ds_timestep_1 and depth == self.ds_depth_1:
            resize_scale = self.resize_scale_1
        elif self.ds_timestep_1 >= timestep > self.ds_timestep_2 and depth == self.ds_depth_2:
            resize_scale = self.resize_scale_2
        else:
            resize_scale = 1

        if resize_scale != 1:
            h = torch.nn.functional.interpolate(h.float(), scale_factor=1 / resize_scale, mode="bicubic", align_corners=False).to(dtype) # bfloat16対応
        
        return h
    
    def apply(self, model, ds_depth_1, ds_depth_2, ds_timestep_1, ds_timestep_2, resize_scale_1, resize_scale_2):
        new_model = model.clone()
        self.ds_depth_1 = ds_depth_1
        self.ds_depth_2 = ds_depth_2
        self.ds_timestep_1 = ds_timestep_1
        self.ds_timestep_2 = ds_timestep_2
        self.resize_scale_1 = resize_scale_1
        self.resize_scale_2 = resize_scale_2

        def apply_model(model_function, kwargs):
            
            xa = kwargs["input"]
            t = kwargs["timestep"]
            c_concat = kwargs["c"].get("c_concat", None)
            c_crossattn = kwargs["c"].get("c_crossattn", None)
            y = kwargs["c"].get("y", None)
            control = kwargs["c"].get("control", None)
            transformer_options = kwargs["c"].get("transformer_options", None)

            # https://github.com/comfyanonymous/ComfyUI/blob/629e4c552cc30a75d2756cbff8095640af3af163/comfy/model_base.py#L51-L69
            sigma = t
            xc = new_model.model.model_sampling.calculate_input(sigma, xa)
            if c_concat is not None:
                xc = torch.cat([xc] + [c_concat], dim=1)

            context = c_crossattn
            dtype = new_model.model.get_dtype()
            xc = xc.to(dtype)
            t = new_model.model.model_sampling.timestep(t).float()
            context = context.to(dtype)
            extra_conds = {}
            for o in kwargs:
                extra = kwargs[o]
                if hasattr(extra, "to"):
                    extra = extra.to(dtype)
                extra_conds[o] = extra

            x = xc
            timesteps = t
            y = None if y is None else y.to(dtype)
            transformer_options["original_shape"] = list(x.shape)
            transformer_options["current_index"] = 0
            transformer_patches = transformer_options.get("patches", {})
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """
            unet = new_model.model.diffusion_model

            # https://github.com/comfyanonymous/ComfyUI/blob/629e4c552cc30a75d2756cbff8095640af3af163/comfy/ldm/modules/diffusionmodules/openaimodel.py#L598-L659

            assert (y is not None) == (
                unet.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(unet.dtype)
            emb = unet.time_embed(t_emb)

            if unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)

            h = x.type(unet.dtype)
            depth = 0
            for id, module in enumerate(unet.input_blocks):
                transformer_options["block"] = ("input", id)
                h = forward_timestep_embed(module, h, emb, context, transformer_options)
                h = apply_control(h, control, 'input')
                hs.append(h)

                # changed
                h = self.hires_resize(h, timesteps[0], depth)
                depth += 1

            transformer_options["block"] = ("middle", 0)
            h = forward_timestep_embed(unet.middle_block, h, emb, context, transformer_options)
            h = apply_control(h, control, 'middle')

            for id, module in enumerate(unet.output_blocks):

                depth -= 1

                transformer_options["block"] = ("output", id)
                hsp = hs.pop()
                hsp = apply_control(hsp, control, 'output')

                # changed
                h = torch.nn.functional.interpolate(h.float(), size=(hsp.shape[2], hsp.shape[3]), mode="bicubic", align_corners=False).to(hsp.dtype) # bfloat16対応
                
                if "output_block_patch" in transformer_patches:
                    patch = transformer_patches["output_block_patch"]
                    for p in patch:
                        h, hsp = p(h, hsp, transformer_options)

                h = th.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape)
            h = h.type(x.dtype)
            if unet.predict_codebook_ids:
                model_output =  unet.id_predictor(h)
            else:
                model_output =  unet.out(h)
            
            return new_model.model.model_sampling.calculate_denoised(sigma, model_output, xa)

        new_model.set_model_unet_function_wrapper(apply_model)

        return (new_model, )


# NODE_CLASS_MAPPINGS = {
#     "Hires": Hires,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Hires": "Apply Kohya's HiresFix",
# }

# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]