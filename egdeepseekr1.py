cpp=print
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from folder_paths import models_dir
from comfy import model_management, model_patcher
import os,sys,random
import numpy as np
from PIL import Image, PngImagePlugin, ImageDraw, ImageFont, ImageColor, ImageChops, ImageOps, ImageFilter
from PIL.Image import fromarray
from torchvision import transforms
import requests
from tqdm import tqdm

deep_model_folder_path = Path(models_dir) / 'deepseek'
deep_model_folder_path.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_URLS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "config.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json",
        "model.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/resolve/main/model.safetensors",
        "tokenizer.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/tokenizer.json",
        "tokenizer_config.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/tokenizer_config.json"
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "config.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/config.json",
        "model-00001-of-000002.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/model-00001-of-000002.safetensors",
        "model-00002-of-000002.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/resolve/main/model-00002-of-000002.safetensors",
        "tokenizer.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/tokenizer.json",
        "tokenizer_config.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/tokenizer_config.json"
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "config.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/raw/main/config.json",
        "model-00001-of-000004.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/model-00001-of-000004.safetensors",
        "model-00002-of-000004.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/model-00002-of-000004.safetensors",
        "model-00003-of-000004.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/model-00003-of-000004.safetensors",
        "model-00004-of-000004.safetensors": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/resolve/main/model-00004-of-000004.safetensors",
        "tokenizer.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/raw/main/tokenizer.json",
        "tokenizer_config.json": "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/raw/main/tokenizer_config.json"
    }
}

MODEL_CONFIGS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": {
        "params": "1.78B",
        "context_length": 32768,
        "tensor_type": "BF16"
    },
    "DeepSeek-R1-Distill-Qwen-7B": {
        "params": "7.62B",
        "context_length": 32768,
        "tensor_type": "BF16"
    },
    "DeepSeek-R1-Distill-Qwen-14B": {
        "params": "14B",
        "context_length": 32768,
        "tensor_type": "BF16"
    }
}

def download_file(url, destination, desc=None):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            print(f"警告: 无法获取文件大小信息")
        
        with open(destination, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        if destination.stat().st_size == 0:
            raise Exception("下载的文件为空")
            
    except Exception as e:
        if destination.exists():
            destination.unlink()
        raise Exception(f"下载失败: {str(e)}")

def ensure_model_downloaded(model_name):
    model_path = deep_model_folder_path / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    if model_name in DEFAULT_MODEL_URLS:
        model_files = DEFAULT_MODEL_URLS[model_name]
        config = MODEL_CONFIGS.get(model_name, {})
        
        print(f"\n正在下载 {model_name} 模型")
        print(f"模型参数量: {config.get('params', 'Unknown')}")
        print(f"上下文长度: {config.get('context_length', 'Unknown')}")
        print(f"张量类型: {config.get('tensor_type', 'Unknown')}\n")
        
        success = True
        for file_name, url in model_files.items():
            file_path = model_path / file_name
            if not file_path.exists() or file_path.stat().st_size == 0:
                try:
                    print(f"正在下载 {file_name}...")
                    download_file(url, file_path, desc=f"Downloading {file_name}")
                    
                    if not file_path.exists() or file_path.stat().st_size == 0:
                        print(f"错误: {file_name} 下载失败或文件为空")
                        success = False
                        break
                except Exception as e:
                    print(f"下载 {file_name} 时出错: {str(e)}")
                    success = False
                    break
        required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        if model_name == "DeepSeek-R1-Distill-Qwen-1.5B":
            required_files.append('model.safetensors')
        elif model_name == "DeepSeek-R1-Distill-Qwen-7B":
            required_files.extend(['model-00001-of-000002.safetensors', 'model-00002-of-000002.safetensors'])
        elif model_name == "DeepSeek-R1-Distill-Qwen-14B":
            required_files.extend([
                'model-00001-of-000004.safetensors',
                'model-00002-of-000004.safetensors',
                'model-00003-of-000004.safetensors',
                'model-00004-of-000004.safetensors'
            ])
            
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists() or file_path.stat().st_size == 0:
                print(f"错误: 缺少必需文件 {file_name} 或文件为空")
                success = False
        
        if not success:
            print(f"\n下载失败。请按照以下步骤手动下载模型文件：")
            print(f"1. 访问模型仓库：https://huggingface.co/deepseek-ai/{model_name}")
            print(f"2. 下载以下文件并放置到 {model_path} 目录：")
            for file_name in required_files:
                print(f"   - {file_name}")
            print(f"3. 确保所有文件下载完整且非空")
            return False
            
        return True
    return False

def deep_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def deep_file_exists(filename, verbose=0):
    filename = str(filename)
    filename = filename.lstrip().rstrip()
    exists = False
    if filename != '':
        my_file = Path(filename)
        exists = my_file.exists()
    if verbose:
        cpp(exists, filename)
    return exists

def deep_read_file(filename, encoding='utf8'):
    if not deep_file_exists(filename):
        return ''
    with open(filename, 'r', encoding=encoding) as f:
        return f.read()

def deep_read_dict(filename, encoding='utf8'):
    content = deep_read_file(filename, encoding=encoding)
    return eval(content)

class DeepModel:
    def __init__(self, model, patcher, tokenizer=None, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.patcher = patcher
        def set_value(self, new_value):
            pass
        model.__class__.device = property(fget=model.__class__.device.fget, fset=set_value)
    
    def clear_cache(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        if hasattr(self, 'patcher') and self.patcher is not None:
            try:
                self.patcher.model = None
                self.patcher = None
            except:
                pass
        torch.cuda.empty_cache()

class egDeepseekR1:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": ([
                    "DeepSeek-R1-Distill-Qwen-1.5B",
                    "DeepSeek-R1-Distill-Qwen-7B",
                    "DeepSeek-R1-Distill-Qwen-14B"
                ], {
                    "default": "DeepSeek-R1-Distill-Qwen-1.5B",
                    "description": "选择模型大小"
                }),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "placeholder": "设置AI助手的角色和行为规则，例如：\n你是一位AI助手，请用简洁的中文回答问题。"
                }),
                "enable_think": ("BOOLEAN", {
                    "default": True,
                    "description": "启用思考模式，让AI先思考再回答"
                }),
                "max_tokens": ("INT", {"default": 2000, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 101}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
                "cache_mode": (["启用缓存", "禁用缓存", "使用后清除"], {"default": "启用缓存"}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 888, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("response", "chat_history", "only_response",)
    FUNCTION = "deep_xgen"
    CATEGORY = "2🐕/deepseekr1"
    
    class_type = "deep_gen"
    OUTPUT_NODE = False

    def __init__(self):
        self.cached_model = None
        self.cached_model_name = None
        self.chat_history = []

    def parse_history(self, history_text):
        messages = []
        if history_text.strip():
            lines = history_text.strip().split('\n')
            current_role = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("User:"):
                    if current_role:
                        messages.append({"role": current_role, "content": '\n'.join(current_content)})
                    current_role = "user"
                    current_content = [line[5:].strip()]
                elif line.startswith("Assistant:"):
                    if current_role:
                        messages.append({"role": current_role, "content": '\n'.join(current_content)})
                    current_role = "assistant"
                    current_content = [line[10:].strip()]
                elif line:
                    current_content.append(line)
                    
            if current_role:
                messages.append({"role": current_role, "content": '\n'.join(current_content)})
                
        return messages

    def load_model(self, model_name, use_cache=True):
        try:
            if use_cache and self.cached_model and self.cached_model_name == model_name:
                if (hasattr(self.cached_model, 'model') and 
                    self.cached_model.model is not None and 
                    hasattr(self.cached_model, 'tokenizer') and 
                    self.cached_model.tokenizer is not None):
                    return self.cached_model
            if self.cached_model:
                self.cached_model.clear_cache()
                self.cached_model = None
                
            offload_device = torch.device('cuda')
            load_device = model_management.get_torch_device()
            mymod = deep_model_folder_path / model_name
            if not (mymod / "config.json").exists():
                if not ensure_model_downloaded(model_name):
                    raise RuntimeError(f"模型 {model_name} 不存在且无法自动下载")
            
            model = AutoModelForCausalLM.from_pretrained(
                mymod,
                device_map=offload_device, 
                torch_dtype="auto", 
            )
            tokenizer = AutoTokenizer.from_pretrained(mymod)
            patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
            
            deep_model = DeepModel(model, patcher, tokenizer=tokenizer)
            if use_cache:
                self.cached_model = deep_model
                self.cached_model_name = model_name
            
            return deep_model
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def format_chat_history(self, messages):
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    def extract_last_response(self, response, enable_think=True):
        if not response:
            return response
        split_position = response.rfind('</think>')
        if split_position == -1:
            processed_text = '\n'.join(line.strip() for line in response.split('\n') if line.strip())
            return processed_text
        extracted_text = response[split_position + 8:].strip()
        processed_text = '\n'.join(line.strip() for line in extracted_text.split('\n') if line.strip())
        
        return processed_text

    def deep_xgen(self, model_size, user_prompt, system_prompt="", 
                 enable_think=True, seed=0, temperature=1.0, max_tokens=500, 
                 top_k=50, top_p=1.0, cache_mode="启用缓存", clear_history=False):
        try:
            use_cache = (cache_mode == "启用缓存")
            clear_after = (cache_mode == "使用后清除")
            if clear_history:
                self.chat_history = []
            
            deep_model = self.load_model(model_size, use_cache)
            
            deep_set_seed(seed % 9999999)
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_prompt})
            
            tokenizer = deep_model.tokenizer
            model = deep_model.model
            patcher = deep_model.patcher
            
            model_management.load_model_gpu(patcher)
            if enable_think:
                text = "<think>\n" + tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.chat_history.append({"role": "user", "content": user_prompt})
            self.chat_history.append({"role": "assistant", "content": response})
            chat_history_text = self.format_chat_history(messages + [{"role": "assistant", "content": response}])
            only_response = self.extract_last_response(response, enable_think)
            
            if clear_after:
                deep_model.clear_cache()
                torch.cuda.empty_cache()
                self.cached_model = None
                self.cached_model_name = None
                
            return (response, chat_history_text, only_response,)
            
        except Exception as e:
            print(f"生成文本时出错: {str(e)}")
            return ("生成失败: " + str(e), "", "",)
