import os
import requests
import torch
import folder_paths
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from groq import Groq
from .image import MS_Image_v2 as MS_Image

from ...utils.log import log

# class MS_Llm_NYU():
#     # https://github.com/cambrian-mllm/cambrian/blob/main/inference.py

class MS_Llm_Microsoft():

    @classmethod
    def __init__(self, model_name = "microsoft/Florence-2-large"):
        self.name = model_name
        if model_name == 'microsoft/kosmos-2-patch14-224':
            self.model = AutoModelForVision2Seq.from_pretrained(self.name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.name)
        self.processor = AutoProcessor.from_pretrained(self.name)
        
    @classmethod
    def generate_prompt(self, image):
        
        _image = MS_Image.tensor2pil(image)
        
        # Generate the caption
        if self.name == 'microsoft/kosmos-2-patch14-224':
            prompt_prefix = "<grounding>"
            inputs = self.processor(text=prompt_prefix, images=_image, return_tensors="pt")
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            caption, _ = self.processor.post_process_generation(generated_text)

        else:
            prompt = "<MORE_DETAILED_CAPTION>"            
            inputs = self.processor(text=prompt, images=_image, return_tensors="pt")
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            caption = self.processor.post_process_generation(generated_text, task=prompt, image_size=(_image.width, _image.height))
            
        return caption


class MS_Llm_Salesforce():
    
    @classmethod
    def __init__(self, model_name = "Salesforce/blip-image-captioning-large"):
        self.name = model_name
        self.model = BlipForConditionalGeneration.from_pretrained(self.name)
        self.processor = BlipProcessor.from_pretrained(self.name)
                
    @classmethod
    def generate_prompt(self, image):
        
        prompt_prefix = "<grounding>"
        
        _image = MS_Image.tensor2pil(image)
        
        inputs = self.processor(text=prompt_prefix, images=_image, return_tensors="pt")

        # Generate the caption
        generated_ids = self.model.generate(**inputs)
        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption

class MS_Llm_Nlpconnect():
    
    @classmethod
    def __init__(self, model_name = "nlpconnect/vit-gpt2-image-captioning"):
        self.name = model_name
        self.model = VisionEncoderDecoderModel.from_pretrained(self.name)
        self.processor = ViTImageProcessor.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)        
                
    @classmethod
    def generate_prompt(self, image):
        
        _image = MS_Image.tensor2pil(image)
        inputs = self.processor(images=_image, return_tensors="pt")
        # Generate the caption
        generated_ids = self.model.generate(
            inputs.pixel_values, 
            max_length=16, 
            num_beams=4, 
            num_return_sequences=1
        )
        caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return caption
    
class MS_Llm():
    
    LLM_MODELS = [
        # 'gemma-7b-it',
        'llama3-70b-8192',
        # 'llama3-8b-8192',
        # 'mixtral-8x7b-32768',
    ]

    # list of model https://huggingface.co/models?pipeline_tag=image-to-text&sort=downloads
    VISION_LLM_MODELS = [
        # 'nlpconnect/vit-gpt2-image-captioning',
        'microsoft/Florence-2-large',
        # 'microsoft/kosmos-2-patch14-224',
        # 'Salesforce/blip-image-captioning-large',
    ]    
    
    @staticmethod
    def prestartup_script():
        folder_paths.add_model_folder_path("nlpconnect", os.path.join(folder_paths.models_dir, "nlpconnect"))
    
    @classmethod
    def __init__(self, vision_llm_name = "nlpconnect/vit-gpt2-image-captioning", llm_name = "llama3-8b-8192"):
        
        if vision_llm_name == 'microsoft/kosmos-2-patch14-224':
            self.vision_llm = MS_Llm_Microsoft()
        elif vision_llm_name == 'Salesforce/blip-image-captioning-large':
            self.vision_llm = MS_Llm_Salesforce()
        else:
            self.vision_llm = MS_Llm_Nlpconnect()

        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self.llm = llm_name

    @classmethod
    def generate_tile_prompt(self, image, prompt_context, seed=None):
        prompt_tile = self.vision_llm.generate_prompt(image)
        if self.vision_llm.name == 'microsoft/kosmos-2-patch14-224':
            _prompt = self.get_grok_prompt(prompt_context, prompt_tile)
        else:
            _prompt = self.get_grok_prompt(prompt_context, prompt_tile)
        if self._groq_key != "":
            prompt = self.call_grok_api(_prompt, seed)
        else:
            prompt = _prompt
        log(prompt, None, None, self.vision_llm.name)
        return prompt

        
    @classmethod
    def get_grok_prompt(self, prompt_context, prompt_tile):
        prompt = [
            f"tile_prompt: \"{prompt_tile}\".",
            f"full_image_prompt: \"{prompt_context}\".",
            "tile_prompt is part of full_image_prompt.",
            "If tile_prompt is describing something different than the full image, correct tile_prompt to match full_image_prompt.",
            "if you don't need to change the tile_prompt return the tile_prompt.",
            "your answer will strictly and only return the tile_prompt string without any decoration like markdown syntax."
        ]
        return " ".join(prompt)
    
    @classmethod
    def call_grok_api(self, prompt, seed=None):

        client = Groq(api_key=self._groq_key)  # Assuming the Groq client accepts an api_key parameter
        completion = client.chat.completions.create(
            model=self.llm,
            messages=[{
                    "role": "user",
                    "content": prompt
            }],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
            seed=seed,
        )

        return completion.choices[0].message.content
