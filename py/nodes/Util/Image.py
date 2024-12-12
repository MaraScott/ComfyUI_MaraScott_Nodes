from comfy_extras.nodes_post_processing import Blur
from ...vendor.ComfyUI_KJNodes.nodes.image_nodes import ColorMatch as ColorMatch


from ...utils.constants import get_name, get_category
from ...utils.log import log

class ImageToGradient_v1:
    
    NAME = get_name('Image To Gradient', "i")    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("gradient",)
    FUNCTION = "fn"

    CATEGORY = get_category("Utils")
    
    COLOR_MATCH_METHOD = 'mkl'
    STRENGTH = 0.25

    def fn(self, image):
        
        _image = image
        for i in range(33):
            _image = Blur().blur(image=_image, blur_radius=31, sigma=1.0)[0]
            
        gradient = ColorMatch().colormatch(image, _image, self.COLOR_MATCH_METHOD, self.STRENGTH)[0]
        
        return (gradient,)