import os
import torch
from transformers import BitsAndBytesConfig
from auto_gptq import BaseQuantizeConfig

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    
    config.tokenizer_paths=['TheBloke/vicuna-7B-v1.5-GPTQ']
    config.model_paths=['TheBloke/vicuna-7B-v1.5-GPTQ']
    config.batch_size = 128
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     # bnb_4bit_use_double_quant=True,
    #     bnb_4bit_use_double_quant=False, # save an additional 0.4 bits per parameter.
    #     bnb_4bit_quant_type="fp4",
    #     bnb_4bit_compute_dtype=torch.float16 # must match torch_dtype
    # )
    # config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False, "quantization_config" : bnb_config}]
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        model_name_or_path="TheBloke/vicuna-7B-v1.5-GPTQ",
        model_file_base_name="model"
    )
    config.model_kwargs=[{
        "use_safetensors":True,
        "trust_remote_code":False,
        "device":"cuda:0",
        "use_triton":False,
        "quantize_config":quantize_config
    }]
    
    return config