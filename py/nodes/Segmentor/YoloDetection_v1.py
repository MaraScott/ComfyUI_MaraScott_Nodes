import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os
import requests
from tqdm import tqdm

from ...utils.constants import get_name, get_category
from ...utils.log import log


from comfy_extras.nodes_images import ImageCrop

items = [
    'airplane', 
    'apple', 
    'backpack',  
    'banana', 
    'baseball bat', 
    'baseball glove', 
    'bear',  
    'bed', 
    'bench', 
    'bicycle', 
    'bird', 
    'boat', 
    'book',  
    'bottle', 
    'bowl', 
    'broccoli', 
    'bus', 
    'cake', 
    'car', 
    'carrot',  
    'cat',  
    'cell phone',  
    'chair',  
    'clock',  
    'couch', 
    'cow', 
    'cup', 
    'dining table', 
    'dog', 
    'donut',  
    'elephant', 
    'face',
    'fire hydrant', 
    'fork',  
    'frisbee',  
    'furniture',  
    'giraffe',  
    'hair drier', 
    'handbag', 
    'horse',  
    'hot dog',  
    'keyboard',  
    'kite', 
    'knife',  
    'laptop',  
    'microwave',  
    'motorcycle', 
    'mouse', 
    'orange',  
    'oven',  
    'parking meter', 
    'person',
    'pizza',  
    'potted plant',  
    'refrigerator', 
    'remote', 
    'sandwich',  
    'scissors',  
    'sheep',
    'sink',  
    'skateboard',  
    'skis',  
    'snowboard',  
    'spoon',  
    'sports ball', 
    'stop sign', 
    'suitcase',  
    'surfboard', 
    'teddy bear',  
    'tennis racket',
    'tie', 
    'toaster', 
    'toilet',  
    'toothbrush',
    'traffic light', 
    'train', 
    'truck', 
    'tv',  
    'umbrella', 
    'vase',  
    'wine glass', 
    'zebra'
]


folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)


class YoloDetection_v1:
    
    NAME = get_name("Yolo Detection - segmentor")

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "object": (items, {"default": "face"} ),
                "padding": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ( "BBOX", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", )
    RETURN_NAMES = ( "bounding box", "image yolo detections", "image original crop", "image square crop", "mask original crop", "mask square crop", )
    
    INPUT_IS_LIST = False
    FUNCTION = "fn"
    CATEGORY = get_category("segmentor")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (None,) * len(RETURN_TYPES)
    

    def download_large_file(self,url, filename):
        exists = os.path.exists(filename)
        if exists:
            pass
        else:
            print("downloading "+url + " to "+filename)
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        progress_bar.update(len(chunk)) 
                        f.write(chunk)

    def tensor2rgba(self,t: torch.Tensor) -> torch.Tensor:
        size = t.size()
        if (len(size) < 4):
            return t.unsqueeze(3).repeat(1, 1, 1, 4)
        elif size[3] == 1:
            return t.repeat(1, 1, 1, 4)
        elif size[3] == 3:
            alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
            return torch.cat((t, alpha_tensor), dim=3)
        else:
            return t

    def filterbiggestbox(self,_boxes):
        currentselectedindex = -1
        currentarea = -1

        for idx, box in enumerate(_boxes):

            area = box[2]*box[3]
            #print(area)
            if area > currentarea:
                currentarea = area
                currentselectedindex = idx

        return currentselectedindex
    
    def getIndexForItem(self, _chosenitem):
         for idx, item in enumerate(items):
            if _chosenitem == item:
                return idx 
    
    def getBoxesXYWHForSelectedItem(self, _itemindex,_boxes_cls,_all_boxes):
        
        _xyhw_boxes_for_selected_item = []

        for idx, index in enumerate(_boxes_cls):
            if index == _itemindex:
                _xyhw_boxes_for_selected_item.append(_all_boxes.xywh[idx])
                
        return _xyhw_boxes_for_selected_item

    def fn(self, image,object,padding):

        self.download_large_file('https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt',os.path.join(folder_paths.get_folder_paths("yolov8")[0],'yolov8l.pt'))
        self.download_large_file('https://huggingface.co/spaces/cc1234/stashface/resolve/main/.deepface/weights/yolov8n-face.pt?download=true',os.path.join(folder_paths.get_folder_paths("yolov8")[0],'yolov8n-face.pt'))

        # Convert tensor to numpy array and then to PIL Image
        image_tensor = image
        image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image

        if object == "face":
            model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/yolov8n-face.pt')  # load face model
        else:
            model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/yolov8l.pt')  # load regular yolo model
      
        results = model(image)

        im_array = results[0].plot()
        im = Image.fromarray(im_array[...,::-1])
        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0) 
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

        if results[0].boxes.shape[0] != 0:

            bbox = None

            if object == "face": 
                biggestbox_index = self.filterbiggestbox(results[0].boxes.xywh)
                bbox=results[0].boxes.xywh[biggestbox_index]
            else:
                selectedobjectindex = self.getIndexForItem(object)
                selectedboxes = self.getBoxesXYWHForSelectedItem(selectedobjectindex-1,results[0].boxes.cls,results[0].boxes)
                if len(selectedboxes) != 0:
                    biggestbox_index = self.filterbiggestbox(selectedboxes)
                else:
                    return ([0,0,0,],image_tensor,image_tensor,image_tensor,image_tensor,image_tensor)
                
                bbox=selectedboxes[biggestbox_index]

            center_x = bbox[0]
            center_y = bbox[1]
            w = bbox[2]+padding
            h = bbox[3]+padding
            x = center_x - (w/2.0)
            y = center_y - (h/2.0)

            origw = bbox[2]
            origh = bbox[3]
            origx = center_x - (origw/2.0)
            origy = center_y - (origh/2.0)
            
            _original_cropper = ImageCrop()
            _original_cropper_d = _original_cropper.crop(image_tensor,int(origw.item()),int(origh.item()),int(origx.item()),int(origy.item()))
            
            #create the mask
            imagewidth = image_tensor.shape[2]
            imageheight= image_tensor.shape[1]

            mask_orginal = torch.zeros((imageheight, imagewidth))
            mask_orginal[int(origy.item()):int(origy.item()+origh.item())+1, int(origx.item()):int(origx.item()+origw.item())+1] = 1

            if w>h:
                #height has to change
                #how much
                newycoordinate = y-((w-h)/2.0)
                w = w
                h = w
                x = x
                y = newycoordinate
            else:
                newxxoordinate = x-((h-w)/2.0)
                w = h
                h = h
                x = newxxoordinate
                y = y

            if x<0:
                x=torch.tensor(0)

            if y<0:
                y=torch.tensor(0)
        
            cropper = ImageCrop()
            cropped = cropper.crop(image_tensor,int(w.item()),int(h.item()),int(x.item()),int(y.item()))

            
            mask = torch.zeros((imageheight, imagewidth))
            mask[int(y.item()):int(y.item()+h.item())+1, int(x.item()):int(x.item()+w.item())+1] = 1
        
            return ([int(x.item()),int(y.item()),int(w.item()),int(h.item())],image_tensor_out,_original_cropper_d[0],cropped[0],mask_orginal,mask)
        else:
            return ([0,0,0,],image_tensor,image_tensor,image_tensor,image_tensor,image_tensor)
