import os
# Prevent transformers from trying to load TensorFlow/Flax at import time
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# Use a pipeline as a high-level helper
from PIL import Image
import torchvision.transforms as T
import torch
import numpy
import math
import json

# Lazy-initialized model + processor to avoid heavy imports at module load
_nsfw_model = None
_nsfw_processor = None


def tensor2pil(image):
    return Image.fromarray(numpy.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(numpy.uint8))


def pil2tensor(image):
    return torch.from_numpy(numpy.array(image).astype(numpy.float32) / 255.0).unsqueeze(0)


class NSFWDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "nsfw_threshold"}),
                "alternative_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE","BOOLEAN", "STRING",)

    FUNCTION = "run"

    CATEGORY = "NSFWDetection"

    def run(self, image, score, alternative_image):
        transform = T.ToPILImage()
        global _nsfw_model, _nsfw_processor
        if _nsfw_model is None or _nsfw_processor is None:
            # Import transformers lazily to avoid optional TF/Flax dependencies
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            _nsfw_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
            _nsfw_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
        for i in range(len(image)):
            pil_img = transform(image[i].permute(2, 0, 1))
            with torch.no_grad():
                inputs = _nsfw_processor(images=pil_img, return_tensors="pt")
                outputs = _nsfw_model(**{k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)})
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                id2label = getattr(_nsfw_model.config, "id2label", None) or {i: str(i) for i in range(probs.shape[-1])}
                result = [
                    {"label": id2label[idx], "score": float(probs[idx].item())}
                    for idx in range(probs.shape[-1])
                ]
                result.sort(key=lambda x: x["score"], reverse=True)
            image_size = image[i].size()
            width, height = image_size[1], image_size[0]
            for r in result:
                if r["label"] == "nsfw":
                    safe = True if r["score"] <= score else False
                    if not safe:
                        image[i] = pil2tensor(transform(alternative_image[0].permute(2, 0, 1)).resize((width, height), resample=Image.Resampling(2)))

        json_obj = {"score": {
            item["label"]: f"{math.floor(item['score'] * 1000) / 1000:.3f}"
            for item in result
            if "label" in item and "score" in item
        }}

        return (image, safe, json.dumps(json_obj, indent=2),)
