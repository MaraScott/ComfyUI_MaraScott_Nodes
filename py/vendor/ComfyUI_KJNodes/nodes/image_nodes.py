# import numpy as np
# import time
import torch
# import torch.nn.functional as F
# import random
# import math
# import os
# import re
# import json
# import hashlib
# from PIL import ImageGrab, ImageDraw, ImageFont, Image, ImageSequence, ImageOps

# from nodes import MAX_RESOLUTION, SaveImage
# from comfy_extras.nodes_mask import ImageCompositeMasked
# from comfy.cli_args import args
# from comfy.utils import ProgressBar, common_upscale
# import folder_paths
# import model_management

# script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# class ImagePass:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#             },
#         }
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "passthrough"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Passes the image through without modifying it.
# """

#     def passthrough(self, image):
#         return image,

class ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'mkl'
            }),
                
            },
        }
    
    CATEGORY = "KJNodes/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch"
    DESCRIPTION = """
color-matcher enables color transfer across images which comes in handy for automatic  
color-grading of photographs, paintings and film sequences as well as light-field  
and stopmotion corrections.  

The methods behind the mappings are based on the approach from Reinhard et al.,  
the Monge-Kantorovich Linearization (MKL) as proposed by Pitie et al. and our analytical solution  
to a Multi-Variate Gaussian Distribution (MVGD) transfer in conjunction with classical histogram   
matching. As shown below our HM-MVGD-HM compound outperforms existing methods.   
https://github.com/hahnec/color-matcher/

"""
    
    def colormatch(self, image_ref, image_target, method):
        try:
            from color_matcher import ColorMatcher
        except:
            raise Exception("Can't import color-matcher, did you install requirements.txt? Manual install: pip install color-matcher")
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            out.append(torch.from_numpy(image_result))
            
        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)
    
# class SaveImageWithAlpha:
#     def __init__(self):
#         self.output_dir = folder_paths.get_output_directory()
#         self.type = "output"
#         self.prefix_append = ""

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": 
#                     {"images": ("IMAGE", ),
#                     "mask": ("MASK", ),
#                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
#                 "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#                 }

#     RETURN_TYPES = ()
#     FUNCTION = "save_images_alpha"
#     OUTPUT_NODE = True
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Saves an image and mask as .PNG with the mask as the alpha channel. 
# """

#     def save_images_alpha(self, images, mask, filename_prefix="ComfyUI_image_with_alpha", prompt=None, extra_pnginfo=None):
#         from PIL.PngImagePlugin import PngInfo
#         filename_prefix += self.prefix_append
#         full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
#         results = list()
#         if mask.dtype == torch.float16:
#             mask = mask.to(torch.float32)
#         def file_counter():
#             max_counter = 0
#             # Loop through the existing files
#             for existing_file in os.listdir(full_output_folder):
#                 # Check if the file matches the expected format
#                 match = re.fullmatch(fr"{filename}_(\d+)_?\.[a-zA-Z0-9]+", existing_file)
#                 if match:
#                     # Extract the numeric portion of the filename
#                     file_counter = int(match.group(1))
#                     # Update the maximum counter value if necessary
#                     if file_counter > max_counter:
#                         max_counter = file_counter
#             return max_counter

#         for image, alpha in zip(images, mask):
#             i = 255. * image.cpu().numpy()
#             a = 255. * alpha.cpu().numpy()
#             img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
#              # Resize the mask to match the image size
#             a_resized = Image.fromarray(a).resize(img.size, Image.LANCZOS)
#             a_resized = np.clip(a_resized, 0, 255).astype(np.uint8)
#             img.putalpha(Image.fromarray(a_resized, mode='L'))
#             metadata = None
#             if not args.disable_metadata:
#                 metadata = PngInfo()
#                 if prompt is not None:
#                     metadata.add_text("prompt", json.dumps(prompt))
#                 if extra_pnginfo is not None:
#                     for x in extra_pnginfo:
#                         metadata.add_text(x, json.dumps(extra_pnginfo[x]))
           
#             # Increment the counter by 1 to get the next available value
#             counter = file_counter() + 1
#             file = f"{filename}_{counter:05}.png"
#             img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
#             results.append({
#                 "filename": file,
#                 "subfolder": subfolder,
#                 "type": self.type
#             })

#         return { "ui": { "images": results } }

# class ImageConcanate:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "image1": ("IMAGE",),
#             "image2": ("IMAGE",),
#             "direction": (
#             [   'right',
#                 'down',
#                 'left',
#                 'up',
#             ],
#             {
#             "default": 'right'
#              }),
#             "match_image_size": ("BOOLEAN", {"default": False}),
#         }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "concanate"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Concatenates the image2 to image1 in the specified direction.
# """

#     def concanate(self, image1, image2, direction, match_image_size):
#         # Check if the batch sizes are different
#         batch_size1 = image1.size(0)
#         batch_size2 = image2.size(0)
        
#         if batch_size1 != batch_size2:
#             # Calculate the number of repetitions needed
#             max_batch_size = max(batch_size1, batch_size2)
#             repeats1 = max_batch_size // batch_size1
#             repeats2 = max_batch_size // batch_size2
            
#             # Repeat the images to match the largest batch size
#             image1 = image1.repeat(repeats1, 1, 1, 1)
#             image2 = image2.repeat(repeats2, 1, 1, 1)
#         if match_image_size:
#             image2 = torch.nn.functional.interpolate(image2, size=(image1.shape[2], image1.shape[3]), mode="bilinear")
#         if direction == 'right':
#             row = torch.cat((image1, image2), dim=2)
#         elif direction == 'down':
#             row = torch.cat((image1, image2), dim=1)
#         elif direction == 'left':
#             row = torch.cat((image2, image1), dim=2)
#         elif direction == 'up':
#             row = torch.cat((image2, image1), dim=1)
#         return (row,)
    
# class ImageGridComposite2x2:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "image1": ("IMAGE",),
#             "image2": ("IMAGE",),
#             "image3": ("IMAGE",),
#             "image4": ("IMAGE",),   
#         }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "compositegrid"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Concatenates the 4 input images into a 2x2 grid. 
# """

#     def compositegrid(self, image1, image2, image3, image4):
#         top_row = torch.cat((image1, image2), dim=2)
#         bottom_row = torch.cat((image3, image4), dim=2)
#         grid = torch.cat((top_row, bottom_row), dim=1)
#         return (grid,)
    
# class ImageGridComposite3x3:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "image1": ("IMAGE",),
#             "image2": ("IMAGE",),
#             "image3": ("IMAGE",),
#             "image4": ("IMAGE",),
#             "image5": ("IMAGE",),
#             "image6": ("IMAGE",),
#             "image7": ("IMAGE",),
#             "image8": ("IMAGE",),
#             "image9": ("IMAGE",),     
#         }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "compositegrid"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Concatenates the 9 input images into a 3x3 grid. 
# """

#     def compositegrid(self, image1, image2, image3, image4, image5, image6, image7, image8, image9):
#         top_row = torch.cat((image1, image2, image3), dim=2)
#         mid_row = torch.cat((image4, image5, image6), dim=2)
#         bottom_row = torch.cat((image7, image8, image9), dim=2)
#         grid = torch.cat((top_row, mid_row, bottom_row), dim=1)
#         return (grid,)
    
# class ImageBatchTestPattern:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "batch_size": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
#             "start_from": ("INT", {"default": 0,"min": 0, "max": 255, "step": 1}),
#             "text_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
#             "text_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
#             "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
#             "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
#             "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
#             "font_size": ("INT", {"default": 255,"min": 8, "max": 4096, "step": 1}),
#         }}

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "generatetestpattern"
#     CATEGORY = "KJNodes/text"

#     def generatetestpattern(self, batch_size, font, font_size, start_from, width, height, text_x, text_y):
#         out = []
#         # Generate the sequential numbers for each image
#         numbers = np.arange(start_from, start_from + batch_size)
#         font_path = folder_paths.get_full_path("kjnodes_fonts", font)

#         for number in numbers:
#             # Create a black image with the number as a random color text
#             image = Image.new("RGB", (width, height), color='black')
#             draw = ImageDraw.Draw(image)
            
#             # Generate a random color for the text
#             font_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
#             font = ImageFont.truetype(font_path, font_size)
            
#             # Get the size of the text and position it in the center
#             text = str(number)
           
#             try:
#                 draw.text((text_x, text_y), text, font=font, fill=font_color, features=['-liga'])
#             except:
#                 draw.text((text_x, text_y), text, font=font, fill=font_color,)
            
#             # Convert the image to a numpy array and normalize the pixel values
#             image_np = np.array(image).astype(np.float32) / 255.0
#             image_tensor = torch.from_numpy(image_np).unsqueeze(0)
#             out.append(image_tensor)
#         out_tensor = torch.cat(out, dim=0)
  
#         return (out_tensor,)

# class ImageGrabPIL:

#     @classmethod
#     def IS_CHANGED(cls):

#         return

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("image",)
#     FUNCTION = "screencap"
#     CATEGORY = "KJNodes/experimental"
#     DESCRIPTION = """
# Captures an area specified by screen coordinates.  
# Can be used for realtime diffusion with autoqueue.
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "x": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
#                  "y": ("INT", {"default": 0,"min": 0, "max": 4096, "step": 1}),
#                  "width": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
#                  "height": ("INT", {"default": 512,"min": 0, "max": 4096, "step": 1}),
#                  "num_frames": ("INT", {"default": 1,"min": 1, "max": 255, "step": 1}),
#                  "delay": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 10.0, "step": 0.01}),
#         },
#     } 

#     def screencap(self, x, y, width, height, num_frames, delay):
#         captures = []
#         bbox = (x, y, x + width, y + height)
        
#         for _ in range(num_frames):
#             # Capture screen
#             screen_capture = ImageGrab.grab(bbox=bbox)
#             screen_capture_torch = torch.tensor(np.array(screen_capture), dtype=torch.float32) / 255.0
#             screen_capture_torch = screen_capture_torch.unsqueeze(0)
#             captures.append(screen_capture_torch)
            
#             # Wait for a short delay if more than one frame is to be captured
#             if num_frames > 1:
#                 time.sleep(delay)
        
#         return (torch.cat(captures, dim=0),)
    
# class AddLabel:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "image":("IMAGE",),  
#             "text_x": ("INT", {"default": 10, "min": 0, "max": 4096, "step": 1}),
#             "text_y": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1}),
#             "height": ("INT", {"default": 48, "min": 0, "max": 4096, "step": 1}),
#             "font_size": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1}),
#             "font_color": ("STRING", {"default": "white"}),
#             "label_color": ("STRING", {"default": "black"}),
#             "font": (folder_paths.get_filename_list("kjnodes_fonts"), ),
#             "text": ("STRING", {"default": "Text"}),
#             "direction": (
#             [   'up',
#                 'down',
#                 'left',
#                 'right',
#                 'overlay'
#             ],
#             {
#             "default": 'up'
#              }),
#             },
#             "optional":{
#                 "caption": ("STRING", {"default": "", "forceInput": True}),
#             }
#             }
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "addlabel"
#     CATEGORY = "KJNodes/text"
#     DESCRIPTION = """
# Creates a new with the given text, and concatenates it to  
# either above or below the input image.  
# Note that this changes the input image's height!  
# Fonts are loaded from this folder:  
# ComfyUI/custom_nodes/ComfyUI-KJNodes/fonts
# """
        
#     def addlabel(self, image, text_x, text_y, text, height, font_size, font_color, label_color, font, direction, caption=""):
#         batch_size = image.shape[0]
#         width = image.shape[2]
        
#         font_path = os.path.join(script_directory, "fonts", "TTNorms-Black.otf") if font == "TTNorms-Black.otf" else folder_paths.get_full_path("kjnodes_fonts", font)
        
#         def process_image(input_image, caption_text):
#             if direction == 'overlay':
#                 pil_image = Image.fromarray((input_image.cpu().numpy() * 255).astype(np.uint8))
#             else:
#                 label_image = Image.new("RGB", (width, height), label_color)
#                 pil_image = label_image
                
#             draw = ImageDraw.Draw(pil_image)
#             font = ImageFont.truetype(font_path, font_size)
            
#             words = caption_text.split()
            
#             lines = []
#             current_line = []
#             current_line_width = 0
#             for word in words:
#                 word_width = font.getbbox(word)[2]
#                 if current_line_width + word_width <= width - 2 * text_x:
#                     current_line.append(word)
#                     current_line_width += word_width + font.getbbox(" ")[2] # Add space width
#                 else:
#                     lines.append(" ".join(current_line))
#                     current_line = [word]
#                     current_line_width = word_width
            
#             if current_line:
#                 lines.append(" ".join(current_line))
            
#             y_offset = text_y
#             for line in lines:
#                 try:
#                     draw.text((text_x, y_offset), line, font=font, fill=font_color, features=['-liga'])
#                 except:
#                     draw.text((text_x, y_offset), line, font=font, fill=font_color)
#                 y_offset += font_size # Move to the next line
                
#             processed_image = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
#             return processed_image
        
#         if caption == "":
#             processed_images = [process_image(img, text) for img in image]
#         else:
#             assert len(caption) == batch_size, "Number of captions does not match number of images"
#             processed_images = [process_image(img, cap) for img, cap in zip(image, caption)]
#         processed_batch = torch.cat(processed_images, dim=0)

#         # Combine images based on direction
#         if direction == 'down':
#             combined_images = torch.cat((image, processed_batch), dim=1)
#         elif direction == 'up':
#             combined_images = torch.cat((processed_batch, image), dim=1)
#         elif direction == 'left':
#             processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
#             combined_images = torch.cat((processed_batch, image), dim=2)
#         elif direction == 'right':
#             processed_batch = torch.rot90(processed_batch, 3, (2, 3)).permute(0, 3, 1, 2)
#             combined_images = torch.cat((image, processed_batch), dim=2)
#         else:
#             combined_images = processed_batch
        
#         return (combined_images,)
    
# class GetImageSizeAndCount:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#             "image": ("IMAGE",),
#         }}

#     RETURN_TYPES = ("IMAGE","INT", "INT", "INT",)
#     RETURN_NAMES = ("image", "width", "height", "count",)
#     FUNCTION = "getsize"
#     CATEGORY = "KJNodes/masking"
#     DESCRIPTION = """
# Returns width, height and batch size of the image,  
# and passes it through unchanged.  

# """

#     def getsize(self, image):
#         width = image.shape[2]
#         height = image.shape[1]
#         count = image.shape[0]
#         return {"ui": {
#             "text": [f"{count}x{width}x{height}"]}, 
#             "result": (image, width, height, count) 
#         }
    
# class ImageBatchRepeatInterleaving:
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "repeat"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Repeats each image in a batch by the specified number of times.  
# Example batch of 5 images: 0, 1 ,2, 3, 4  
# with repeats 2 becomes batch of 10 images: 0, 0, 1, 1, 2, 2, 3, 3, 4, 4  
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "images": ("IMAGE",),
#                  "repeats": ("INT", {"default": 1, "min": 1, "max": 4096}),
#         },
#     } 
    
#     def repeat(self, images, repeats):
       
#         repeated_images = torch.repeat_interleave(images, repeats=repeats, dim=0)
#         return (repeated_images, )
    
# class ImageUpscaleWithModelBatched:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { "upscale_model": ("UPSCALE_MODEL",),
#                               "images": ("IMAGE",),
#                               "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
#                               }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "upscale"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Same as ComfyUI native model upscaling node,  
# but allows setting sub-batches for reduced VRAM usage.
# """
#     def upscale(self, upscale_model, images, per_batch):
        
#         device = model_management.get_torch_device()
#         upscale_model.to(device)
#         in_img = images.movedim(-1,-3).to(device)
        
#         steps = in_img.shape[0]
#         pbar = ProgressBar(steps)
#         t = []
        
#         for start_idx in range(0, in_img.shape[0], per_batch):
#             sub_images = upscale_model(in_img[start_idx:start_idx+per_batch])
#             t.append(sub_images.cpu())
#             # Calculate the number of images processed in this batch
#             batch_count = sub_images.shape[0]
#             # Update the progress bar by the number of images processed in this batch
#             pbar.update(batch_count)
#         upscale_model.cpu()
        
#         t = torch.cat(t, dim=0).permute(0, 2, 3, 1).cpu()

#         return (t,)

# class ImageNormalize_Neg1_To_1:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { 
#                               "images": ("IMAGE",),
    
#                               }}
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "normalize"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Normalize the images to be in the range [-1, 1]  
# """

#     def normalize(self,images):
#         images = images * 2.0 - 1.0
#         return (images,)

# class RemapImageRange:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { 
#             "image": ("IMAGE",),
#             "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
#             "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
#             "clamp": ("BOOLEAN", {"default": True}),
#             },
#             }
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "remap"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Remaps the image values to the specified range. 
# """
        
#     def remap(self, image, min, max, clamp):
#         if image.dtype == torch.float16:
#             image = image.to(torch.float32)
#         image = min + image * (max - min)
#         if clamp:
#             image = torch.clamp(image, min=0.0, max=1.0)
#         return (image, )

# class SplitImageChannels:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { 
#             "image": ("IMAGE",),
#             },
#             }
    
#     RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
#     RETURN_NAMES = ("red", "green", "blue", "mask")
#     FUNCTION = "split"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Splits image channels into images where the selected channel  
# is repeated for all channels, and the alpha as a mask. 
# """
        
#     def split(self, image):
#         red = image[:, :, :, 0:1] # Red channel
#         green = image[:, :, :, 1:2] # Green channel
#         blue = image[:, :, :, 2:3] # Blue channel
#         alpha = image[:, :, :, 3:4] # Alpha channel
#         alpha = alpha.squeeze(-1)

#         # Repeat the selected channel for all channels
#         red = torch.cat([red, red, red], dim=3)
#         green = torch.cat([green, green, green], dim=3)
#         blue = torch.cat([blue, blue, blue], dim=3)
#         return (red, green, blue, alpha)
    
# class MergeImageChannels:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": { 
#             "red": ("IMAGE",),
#             "green": ("IMAGE",),
#             "blue": ("IMAGE",),
            
#             },
#             "optional": {
#                 "mask": ("MASK", {"default": None}),
#                 },
#             }
    
#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("image",)
#     FUNCTION = "merge"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Merges channel data into an image.  
# """
        
#     def merge(self, red, green, blue, alpha=None):
#         image = torch.stack([
#         red[..., 0, None], # Red channel
#         green[..., 1, None], # Green channel
#         blue[..., 2, None]   # Blue channel
#         ], dim=-1)
#         image = image.squeeze(-2)
#         if alpha is not None:
#             image = torch.cat([image, alpha], dim=-1)
#         return (image,)

# class ImagePadForOutpaintMasked:

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
#                 "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
#                 "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
#                 "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
#                 "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
#             },
#             "optional": {
#                 "mask": ("MASK",),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "MASK")
#     FUNCTION = "expand_image"

#     CATEGORY = "image"

#     def expand_image(self, image, left, top, right, bottom, feathering, mask=None):
#         if mask is not None:
#             if torch.allclose(mask, torch.zeros_like(mask)):
#                     print("Warning: The incoming mask is fully black. Handling it as None.")
#                     mask = None
#         B, H, W, C = image.size()

#         new_image = torch.ones(
#             (B, H + top + bottom, W + left + right, C),
#             dtype=torch.float32,
#         ) * 0.5

#         new_image[:, top:top + H, left:left + W, :] = image

#         if mask is None:
#             new_mask = torch.ones(
#                 (B, H + top + bottom, W + left + right),
#                 dtype=torch.float32,
#             )

#             t = torch.zeros(
#             (B, H, W),
#             dtype=torch.float32
#             )
#         else:
#             # If a mask is provided, pad it to fit the new image size
#             mask = F.pad(mask, (left, right, top, bottom), mode='constant', value=0)
#             mask = 1 - mask
#             t = torch.zeros_like(mask)
        
#         if feathering > 0 and feathering * 2 < H and feathering * 2 < W:

#             for i in range(H):
#                 for j in range(W):
#                     dt = i if top != 0 else H
#                     db = H - i if bottom != 0 else H

#                     dl = j if left != 0 else W
#                     dr = W - j if right != 0 else W

#                     d = min(dt, db, dl, dr)

#                     if d >= feathering:
#                         continue

#                     v = (feathering - d) / feathering

#                     if mask is None:
#                         t[:, i, j] = v * v
#                     else:
#                         t[:, top + i, left + j] = v * v
        
#         if mask is None:
#             new_mask[:, top:top + H, left:left + W] = t
#             return (new_image, new_mask,)
#         else:
#             return (new_image, mask,)

# class ImagePadForOutpaintTargetSize:
#     upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
#                 "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
#                 "feathering": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
#                 "upscale_method": (s.upscale_methods,),
#             },
#             "optional": {
#                 "mask": ("MASK",),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "MASK")
#     FUNCTION = "expand_image"

#     CATEGORY = "image"

#     def expand_image(self, image, target_width, target_height, feathering, upscale_method, mask=None):
#         B, H, W, C = image.size()
#         new_height = 0
#         new_width = 0
#          # Calculate the scaling factor while maintaining aspect ratio
#         scaling_factor = min(target_width / W, target_height / H)
        
#         # Check if the image needs to be downscaled
#         if scaling_factor < 1:
#             image = image.movedim(-1,1)
#             # Calculate the new width and height after downscaling
#             new_width = int(W * scaling_factor)
#             new_height = int(H * scaling_factor)
            
#             # Downscale the image
#             image_scaled = common_upscale(image, new_width, new_height, upscale_method, "disabled").movedim(1,-1)
#             if mask is not None:
#                 mask_scaled = mask.unsqueeze(0)  # Add an extra dimension for batch size
#                 mask_scaled = F.interpolate(mask_scaled, size=(new_height, new_width), mode="nearest")
#                 mask_scaled = mask_scaled.squeeze(0)  # Remove the extra dimension after interpolation
#             else:
#                 mask_scaled = mask
#         else:
#             # If downscaling is not needed, use the original image dimensions
#             image_scaled = image
#             mask_scaled = mask

#         # Calculate how much padding is needed to reach the target dimensions
#         pad_top = max(0, (target_height - new_height) // 2)
#         pad_bottom = max(0, target_height - new_height - pad_top)
#         pad_left = max(0, (target_width - new_width) // 2)
#         pad_right = max(0, target_width - new_width - pad_left)

#         # Now call the original expand_image with the calculated padding
#         return ImagePadForOutpaintMasked.expand_image(self, image_scaled, pad_left, pad_top, pad_right, pad_bottom, feathering, mask_scaled)
    
# class ImageAndMaskPreview(SaveImage):
#     def __init__(self):
#         self.output_dir = folder_paths.get_temp_directory()
#         self.type = "temp"
#         self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
#         self.compress_level = 4

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "mask_color": ("STRING", {"default": "255, 255, 255"}),
#                 "pass_through": ("BOOLEAN", {"default": False}),
#              },
#             "optional": {
#                 "image": ("IMAGE",),
#                 "mask": ("MASK",),                
#             },
#             "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
#         }
#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("composite",)
#     FUNCTION = "execute"
#     CATEGORY = "KJNodes"
#     DESCRIPTION = """
# Preview an image or a mask, when both inputs are used  
# composites the mask on top of the image.
# with pass_through on the preview is disabled and the  
# composite is returned from the composite slot instead,  
# this allows for the preview to be passed for video combine  
# nodes for example.
# """

#     def execute(self, mask_opacity, mask_color, pass_through, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
#         if mask is not None and image is None:
#             preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
#         elif mask is None and image is not None:
#             preview = image
#         elif mask is not None and image is not None:
#             mask_adjusted = mask * mask_opacity
#             mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

#             color_list = list(map(int, mask_color.split(', ')))
#             mask_image[:, :, :, 0] = color_list[0] // 255 # Red channel
#             mask_image[:, :, :, 1] = color_list[1] // 255 # Green channel
#             mask_image[:, :, :, 2] = color_list[2] // 255 # Blue channel
            
#             preview, = ImageCompositeMasked.composite(self, image, mask_image, 0, 0, True, mask_adjusted)
#         if pass_through:
#             return (preview, )
#         return(self.save_images(preview, filename_prefix, prompt, extra_pnginfo))
        
# class CrossFadeImages:
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "crossfadeimages"
#     CATEGORY = "KJNodes/image"

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "images_1": ("IMAGE",),
#                  "images_2": ("IMAGE",),
#                  "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out", "bounce", "elastic", "glitchy", "exponential_ease_out"],),
#                  "transition_start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
#                  "transitioning_frames": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
#                  "start_level": ("FLOAT", {"default": 0.0,"min": 0.0, "max": 1.0, "step": 0.01}),
#                  "end_level": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 1.0, "step": 0.01}),
#         },
#     } 
    
#     def crossfadeimages(self, images_1, images_2, transition_start_index, transitioning_frames, interpolation, start_level, end_level):

#         def crossfade(images_1, images_2, alpha):
#             crossfade = (1 - alpha) * images_1 + alpha * images_2
#             return crossfade
#         def ease_in(t):
#             return t * t
#         def ease_out(t):
#             return 1 - (1 - t) * (1 - t)
#         def ease_in_out(t):
#             return 3 * t * t - 2 * t * t * t
#         def bounce(t):
#             if t < 0.5:
#                 return self.ease_out(t * 2) * 0.5
#             else:
#                 return self.ease_in((t - 0.5) * 2) * 0.5 + 0.5
#         def elastic(t):
#             return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))
#         def glitchy(t):
#             return t + 0.1 * math.sin(40 * t)
#         def exponential_ease_out(t):
#             return 1 - (1 - t) ** 4

#         easing_functions = {
#             "linear": lambda t: t,
#             "ease_in": ease_in,
#             "ease_out": ease_out,
#             "ease_in_out": ease_in_out,
#             "bounce": bounce,
#             "elastic": elastic,
#             "glitchy": glitchy,
#             "exponential_ease_out": exponential_ease_out,
#         }

#         crossfade_images = []

#         alphas = torch.linspace(start_level, end_level, transitioning_frames)
#         for i in range(transitioning_frames):
#             alpha = alphas[i]
#             image1 = images_1[i + transition_start_index]
#             image2 = images_2[i + transition_start_index]
#             easing_function = easing_functions.get(interpolation)
#             alpha = easing_function(alpha)  # Apply the easing function to the alpha value

#             crossfade_image = crossfade(image1, image2, alpha)
#             crossfade_images.append(crossfade_image)
            
#         # Convert crossfade_images to tensor
#         crossfade_images = torch.stack(crossfade_images, dim=0)
#         # Get the last frame result of the interpolation
#         last_frame = crossfade_images[-1]
#         # Calculate the number of remaining frames from images_2
#         remaining_frames = len(images_2) - (transition_start_index + transitioning_frames)
#         # Crossfade the remaining frames with the last used alpha value
#         for i in range(remaining_frames):
#             alpha = alphas[-1]
#             image1 = images_1[i + transition_start_index + transitioning_frames]
#             image2 = images_2[i + transition_start_index + transitioning_frames]
#             easing_function = easing_functions.get(interpolation)
#             alpha = easing_function(alpha)  # Apply the easing function to the alpha value

#             crossfade_image = crossfade(image1, image2, alpha)
#             crossfade_images = torch.cat([crossfade_images, crossfade_image.unsqueeze(0)], dim=0)
#         # Append the beginning of images_1
#         beginning_images_1 = images_1[:transition_start_index]
#         crossfade_images = torch.cat([beginning_images_1, crossfade_images], dim=0)
#         return (crossfade_images, )

# class GetImageRangeFromBatch:
    
#     RETURN_TYPES = ("IMAGE", "MASK", )
#     FUNCTION = "imagesfrombatch"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Creates a new batch using images from the input,  
# batch, starting from start_index.
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "images": ("IMAGE",),
#                  "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
#                  "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
#         },
#         "optional": {
#             "masks": ("MASK",),
#         }
#     } 
    
#     def imagesfrombatch(self, images, start_index, num_frames, masks=None):
#         if start_index == -1:
#             start_index = len(images) - num_frames
#         if start_index < 0 or start_index >= len(images):
#             raise ValueError("GetImageRangeFromBatch: Start index is out of range")
#         end_index = start_index + num_frames
#         if end_index > len(images):
#             raise ValueError("GetImageRangeFromBatch: End index is out of range")
#         chosen_images = images[start_index:end_index]
#         chosen_masks = masks[start_index:end_index] if masks is not None else None
#         return (chosen_images, chosen_masks,)
    
# class GetImagesFromBatchIndexed:
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "indexedimagesfrombatch"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Selects and returns the images at the specified indices as an image batch.
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "images": ("IMAGE",),
#                  "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
#         },
#     } 
    
#     def indexedimagesfrombatch(self, images, indexes):
        
#         # Parse the indexes string into a list of integers
#         index_list = [int(index.strip()) for index in indexes.split(',')]
        
#         # Convert list of indices to a PyTorch tensor
#         indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
#         # Select the images at the specified indices
#         chosen_images = images[indices_tensor]
        
#         return (chosen_images,)

# class InsertImagesToBatchIndexed:
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "insertimagesfrombatch"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Inserts images at the specified indices into the original image batch.
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "original_images": ("IMAGE",),
#                 "images_to_insert": ("IMAGE",),
#                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
#             },
#         }
    
#     def insertimagesfrombatch(self, original_images, images_to_insert, indexes):
        
#         # Parse the indexes string into a list of integers
#         index_list = [int(index.strip()) for index in indexes.split(',')]
        
#         # Convert list of indices to a PyTorch tensor
#         indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
#         # Ensure the images_to_insert is a tensor
#         if not isinstance(images_to_insert, torch.Tensor):
#             images_to_insert = torch.tensor(images_to_insert)
        
#         # Insert the images at the specified indices
#         for index, image in zip(indices_tensor, images_to_insert):
#             original_images[index] = image
        
#         return (original_images,)

# class ReplaceImagesInBatch:
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "replace"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Replaces the images in a batch, starting from the specified start index,  
# with the replacement images.
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "original_images": ("IMAGE",),
#                  "replacement_images": ("IMAGE",),
#                  "start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
#         },
#     } 
    
#     def replace(self, original_images, replacement_images, start_index):
#         images = None
#         if start_index >= len(original_images):
#             raise ValueError("GetImageRangeFromBatch: Start index is out of range")
#         end_index = start_index + len(replacement_images)
#         if end_index > len(original_images):
#             raise ValueError("GetImageRangeFromBatch: End index is out of range")
#          # Create a copy of the original_images tensor
#         original_images_copy = original_images.clone()
#         original_images_copy[start_index:end_index] = replacement_images
#         images = original_images_copy
#         return (images, )
    

# class ReverseImageBatch:
    
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "reverseimagebatch"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Reverses the order of the images in a batch.
# """

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                  "images": ("IMAGE",),
#         },
#     } 
    
#     def reverseimagebatch(self, images):
#         reversed_images = torch.flip(images, [0])
#         return (reversed_images, )

# class ImageBatchMulti:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
#                 "image_1": ("IMAGE", ),
#                 "image_2": ("IMAGE", ),
#             },
#     }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("images",)
#     FUNCTION = "combine"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Creates an image batch from multiple images.  
# You can set how many inputs the node has,  
# with the **inputcount** and clicking update.
# """

#     def combine(self, inputcount, **kwargs):
#         from nodes import ImageBatch
#         image_batch_node = ImageBatch()
#         image = kwargs["image_1"]
#         for c in range(1, inputcount):
#             new_image = kwargs[f"image_{c + 1}"]
#             image, = image_batch_node.batch(image, new_image)
#         return (image,)

# class ImageAddMulti:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
#                 "image_1": ("IMAGE", ),
#                 "image_2": ("IMAGE", ),
#                 "blending": (
#                 [   'add',
#                     'subtract',
#                     'multiply',
#                     'difference',
#                 ],
#                 {
#                 "default": 'add'
#                 }),
#             },
#     }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("images",)
#     FUNCTION = "add"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Add blends multiple images together.    
# You can set how many inputs the node has,  
# with the **inputcount** and clicking update.
# """

#     def add(self, inputcount, blending, **kwargs):
#         image = kwargs["image_1"]
#         for c in range(1, inputcount):
#             new_image = kwargs[f"image_{c + 1}"]
#             if blending == "add":
#                 image = torch.add(image * 0.5, new_image * 0.5)
#             elif blending == "subtract":
#                 image = torch.sub(image * 0.5, new_image * 0.5)
#             elif blending == "multiply":
#                 image = torch.mul(image * 0.5, new_image * 0.5)
#             elif blending == "difference":
#                 image = torch.sub(image, new_image)
#         return (image,)    

# class PreviewAnimation:
#     def __init__(self):
#         self.output_dir = folder_paths.get_temp_directory()
#         self.type = "temp"
#         self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
#         self.compress_level = 1

#     methods = {"default": 4, "fastest": 0, "slowest": 6}
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {
#                      "fps": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
#                      },
#                 "optional": {
#                     "images": ("IMAGE", ),
#                     "masks": ("MASK", ),
#                 },
#             }

#     RETURN_TYPES = ()
#     FUNCTION = "preview"
#     OUTPUT_NODE = True
#     CATEGORY = "KJNodes/image"

#     def preview(self, fps, images=None, masks=None):
#         filename_prefix = "AnimPreview"
#         full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
#         results = list()

#         pil_images = []

#         if images is not None and masks is not None:
#             for image in images:
#                 i = 255. * image.cpu().numpy()
#                 img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#                 pil_images.append(img)
#             for mask in masks:
#                 if pil_images: 
#                     mask_np = mask.cpu().numpy()
#                     mask_np = np.clip(mask_np * 255, 0, 255).astype(np.uint8)  # Convert to values between 0 and 255
#                     mask_img = Image.fromarray(mask_np, mode='L')
#                     img = pil_images.pop(0)  # Remove and get the first image
#                     img = img.convert("RGBA")  # Convert base image to RGBA

#                     # Create a new RGBA image based on the grayscale mask
#                     rgba_mask_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
#                     rgba_mask_img.putalpha(mask_img)  # Use the mask image as the alpha channel

#                     # Composite the RGBA mask onto the base image
#                     composited_img = Image.alpha_composite(img, rgba_mask_img)
#                     pil_images.append(composited_img)  # Add the composited image back

#         elif images is not None and masks is None:
#             for image in images:
#                 i = 255. * image.cpu().numpy()
#                 img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
#                 pil_images.append(img)

#         elif masks is not None and images is None:
#             for mask in masks:
#                 mask_np = 255. * mask.cpu().numpy()
#                 mask_img = Image.fromarray(np.clip(mask_np, 0, 255).astype(np.uint8))
#                 pil_images.append(mask_img)
#         else:
#             print("PreviewAnimation: No images or masks provided")
#             return { "ui": { "images": results, "animated": (None,), "text": "empty" }}

#         num_frames = len(pil_images)

#         c = len(pil_images)
#         for i in range(0, c, num_frames):
#             file = f"{filename}_{counter:05}_.webp"
#             pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], lossless=False, quality=80, method=4)
#             results.append({
#                 "filename": file,
#                 "subfolder": subfolder,
#                 "type": self.type
#             })
#             counter += 1

#         animated = num_frames != 1
#         return { "ui": { "images": results, "animated": (animated,), "text": [f"{num_frames}x{pil_images[0].size[0]}x{pil_images[0].size[1]}"] } }
    
# class ImageResizeKJ:
#     upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
#                 "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
#                 "upscale_method": (s.upscale_methods,),
#                 "keep_proportion": ("BOOLEAN", { "default": False }),
#                 "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
#             },
#             "optional" : {
#                 "width_input": ("INT", { "forceInput": True}),
#                 "height_input": ("INT", { "forceInput": True}),
#                 "get_image_size": ("IMAGE",),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "INT", "INT",)
#     RETURN_NAMES = ("IMAGE", "width", "height",)
#     FUNCTION = "resize"
#     CATEGORY = "KJNodes/image"
#     DESCRIPTION = """
# Resizes the image to the specified width and height.  
# Size can be retrieved from the inputs, and the final scale  
# is  determined in this order of importance:  
# - get_image_size  
# - width_input and height_input  
# - width and height widgets  
  
# Keep proportions keeps the aspect ratio of the image, by  
# highest dimension.  
# """

#     def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, width_input=None, height_input=None, get_image_size=None):
#         B, H, W, C = image.shape
#         if width_input:
#             width = width_input
#         if height_input:
#             height = height_input
#         if get_image_size is not None:
#             _, height, width, _ = get_image_size.shape

#         if keep_proportion and get_image_size is None:
#             # If one of the dimensions is zero, calculate it to maintain the aspect ratio
#             if width == 0 and height != 0:
#                 ratio = height / H
#                 width = round(W * ratio)
#             elif height == 0 and width != 0:
#                 ratio = width / W
#                 height = round(H * ratio)
#             elif width != 0 and height != 0:
#                 # Scale based on which dimension is smaller in proportion to the desired dimensions
#                 ratio = min(width / W, height / H)
#                 width = round(W * ratio)
#                 height = round(H * ratio)
#         else:
#             if width == 0:
#                 width = W
#             if height == 0:
#                 height = H
    
#         if divisible_by > 1 and get_image_size is None:
#             width = width - (width % divisible_by)
#             height = height - (height % divisible_by)

#         image = image.movedim(-1,1)
#         scaled = common_upscale(image, width, height, upscale_method, 'disabled')
#         scaled = scaled.movedim(1,-1)

#         return(scaled, scaled.shape[2], scaled.shape[1],)
    
# class LoadAndResizeImage:
#     _color_channels = ["alpha", "red", "green", "blue"]
#     @classmethod
#     def INPUT_TYPES(s):
#         input_dir = folder_paths.get_input_directory()
#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
#         return {"required":
#                     {
#                     "image": (sorted(files), {"image_upload": True}),
#                     "resize": ("BOOLEAN", { "default": False }),
#                     "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
#                     "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
#                     "repeat": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
#                     "keep_proportion": ("BOOLEAN", { "default": False }),
#                     "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
#                     "mask_channel": (s._color_channels, ), 
#                     },
#                 }

#     CATEGORY = "KJNodes/image"
#     RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
#     RETURN_NAMES = ("image", "mask", "width", "height",)
#     FUNCTION = "load_image"

#     def load_image(self, image, resize, width, height, repeat, keep_proportion, divisible_by, mask_channel):
#         image_path = folder_paths.get_annotated_filepath(image)

#         import node_helpers
#         img = node_helpers.pillow(Image.open, image_path)
        
#         output_images = []
#         output_masks = []
#         w, h = None, None

#         excluded_formats = ['MPO']

#         W, H = img.size
#         if resize:
#             if keep_proportion:
#                 ratio = min(width / W, height / H)
#                 width = round(W * ratio)
#                 height = round(H * ratio)
#             else:
#                 if width == 0:
#                     width = W
#                 if height == 0:
#                     height = H

#             if divisible_by > 1:
#                 width = width - (width % divisible_by)
#                 height = height - (height % divisible_by)
#         else:
#             width, height = W, H

#         for i in ImageSequence.Iterator(img):
#             i = node_helpers.pillow(ImageOps.exif_transpose, i)

#             if i.mode == 'I':
#                 i = i.point(lambda i: i * (1 / 255))
#             image = i.convert("RGB")

#             if len(output_images) == 0:
#                 w = image.size[0]
#                 h = image.size[1]
            
#             if image.size[0] != w or image.size[1] != h:
#                 continue
#             if resize:
#                 image = image.resize((width, height), Image.Resampling.BILINEAR)

#             image = np.array(image).astype(np.float32) / 255.0
#             image = torch.from_numpy(image)[None,]
#             mask = None
#             c = mask_channel[0].upper()
#             if c in i.getbands():
#                 if resize:
#                     i = i.resize((width, height), Image.Resampling.BILINEAR)
#                 mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
#                 mask = torch.from_numpy(mask)
#                 if c == 'A':
#                      mask = 1. - mask
#             else:
#                 mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

#             output_images.append(image)
#             output_masks.append(mask.unsqueeze(0))

#         if len(output_images) > 1 and img.format not in excluded_formats:
#             output_image = torch.cat(output_images, dim=0)
#             output_mask = torch.cat(output_masks, dim=0)
#         else:
#             output_image = output_images[0]
#             output_mask = output_masks[0]
#             if repeat > 1:
#                 output_image = output_image.repeat(repeat, 1, 1, 1)
#                 output_mask = output_mask.repeat(repeat, 1, 1)

     
#         return (output_image, output_mask, width, height)
        

#     @classmethod
#     def IS_CHANGED(s, image):
#         image_path = folder_paths.get_annotated_filepath(image)
#         m = hashlib.sha256()
#         with open(image_path, 'rb') as f:
#             m.update(f.read())
#         return m.digest().hex()

#     @classmethod
#     def VALIDATE_INPUTS(s, image):
#         if not folder_paths.exists_annotated_filepath(image):
#             return "Invalid image file: {}".format(image)

#         return True