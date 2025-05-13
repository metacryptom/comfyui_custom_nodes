import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import traceback
import random
import time

    

def gpt_edit_image(image_path, prompt, api_key,endpoint, output_path=None, max_retry=3, retry_interval=5, mask_path=None):
    """
    Edit an image using Azure OpenAI's image editing API.
    
    Args:
        image_path (str): Path to the image to edit
        mask_path (str): Path to the mask image
        prompt (str): Text prompt describing the desired edit
        output_path (str, optional): Path to save the edited image. If None, returns the base64 encoded image
        api_key (str, optional): Azure API key. If None, uses environment variable AZURE_API_KEY
        endpoint (str): Azure OpenAI endpoint URL
        api_version (str): API version to use
        max_retry (int): Maximum number of retry attempts
        retry_interval (int): Time to wait between retries in seconds
        
    Returns:
        str or bool: If output_path is None, returns the base64 encoded image. Otherwise, returns True if successful
    """
    
    for attempt in range(max_retry):
        full_url = f"{endpoint}"
        try:
            # Prepare the multipart form data
            files = {
                'image': ('image.png', open(image_path, 'rb')),
                'prompt': (None, prompt)
            }
            if mask_path:
                files['mask'] = ('mask.png', open(mask_path, 'rb'))
            
            # Make the request
            response = requests.post(
                full_url,
                headers={'api-key': api_key},
                files=files,
                timeout=1200,
            )
            
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the JSON response
            response_json = response.json()
            
            if 'data' not in response_json or len(response_json['data']) == 0 or 'b64_json' not in response_json['data'][0]:
                raise Exception(f"Invalid response format: {response_json}")
            
            b64_image = response_json['data'][0]['b64_json']
            
            # If output path is provided, save the image
            return b64_image
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retry - 1:
                time.sleep(retry_interval)
            else:
                raise Exception(f"Failed to edit image after {max_retry} attempts: {str(e)}")

class GPTImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        # 创建临时实例以加载配置
        temp_instance = cls()
        config = temp_instance.load_config()
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "api_url": ("STRING", {"default": config["api_url"], "multiline": False}),
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "masks": ("MASK",),
                "size": (["1024x1024", "1024x1536", "1536x1024"], {"default": "1024x1536"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "GPT-API"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.node_dir, "gpt_api_config.json")
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def save_config(self, api_key, api_url ):
        """保存API配置到文件"""
        if not api_key or len(api_key) < 10:
            return False
            
        try:
            config = {
                "api_key": api_key,
                "api_url": api_url,
                "saved_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.config_file, "w") as f:
                json.dump(config, f)
            self.log("已保存API配置到节点目录")
            return True
        except Exception as e:
            self.log(f"保存API配置失败: {e}")
            return False
    
    def load_config(self):
        """从文件加载API配置"""
        default_config = {
            "api_key": "",
            "api_url": "API URL",
        }
        
        if not os.path.exists(self.config_file):
            return default_config
            
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                
            # 验证必要字段
            if "api_key" not in config or len(config["api_key"]) < 10:
                self.log("已保存的API密钥无效")
                return default_config
                
            self.log("成功加载已保存的API配置")
            return config
        except Exception as e:
            self.log(f"加载API配置失败: {e}")
            return default_config
    
    
    def generate_empty_image(self, width=512, height=512):
        """生成标准格式的空白RGB图像张量"""
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor

    def convert_tensor_to_local_path(self,image_tensor,max_dimension=1024):
        """将ComfyUI的图像张量转换为本地路径"""
        # 创建临时文件夹
        batch_size = image_tensor.shape[0]
        self.log(f"检测到 {batch_size} 张参考图像")
        temp_dir = os.path.join(self.node_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)    

        temp_image_paths = []
        for i in range(batch_size):
            input_image = image_tensor[i].cpu().numpy()
            # 转换为PIL图像
            input_image = (input_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(input_image)
            
            original_width, original_height = pil_image.width, pil_image.height
            self.log(f"参考图像 {i+1} 原始尺寸: {original_width}x{original_height}")

            # 检查是否需要调整大小
            if original_width > max_dimension or original_height > max_dimension:
                # 计算等比例缩放
                if original_width > original_height:
                    new_width = max_dimension
                    new_height = int(original_height * (max_dimension / original_width))
                else:
                    new_height = max_dimension
                    new_width = int(original_width * (max_dimension / original_height))
                
                # 调整图像大小
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                self.log(f"参考图像 {i+1} 已调整尺寸: {new_width}x{new_height}")
            
            self.log(f"参考图像 {i+1} 处理成功，尺寸: {pil_image.width}x{pil_image.height}")

            # 保存图像到临时文件

            temp_image_path = os.path.join(temp_dir, f"input_{int(time.time())}_{i}.png")
            pil_image.save(temp_image_path)
            temp_image_paths.append(temp_image_path)

        return temp_image_paths
    
    
    def encode_images_to_base64(self, image_tensor, max_dimension=1024, quality=85):
        """将ComfyUI的图像张量(单张或多张)转换为base64编码的列表，并进行压缩处理"""
        try:
            # 确定图像数量
            batch_size = image_tensor.shape[0]
            self.log(f"检测到 {batch_size} 张参考图像")
            
            base64_images = []
            
            # 逐一处理每张图像
            for i in range(batch_size):
                # 获取单张图像
                input_image = image_tensor[i].cpu().numpy()

                # 转换为PIL图像
                input_image = (input_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(input_image)
                
                original_width, original_height = pil_image.width, pil_image.height
                self.log(f"参考图像 {i+1} 原始尺寸: {original_width}x{original_height}")
                
                # 检查是否需要调整大小
                if original_width > max_dimension or original_height > max_dimension:
                    # 计算等比例缩放
                    if original_width > original_height:
                        new_width = max_dimension
                        new_height = int(original_height * (max_dimension / original_width))
                    else:
                        new_height = max_dimension
                        new_width = int(original_width * (max_dimension / original_height))
                    
                    # 调整图像大小
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                    self.log(f"参考图像 {i+1} 已调整尺寸: {new_width}x{new_height}")
                
                self.log(f"参考图像 {i+1} 处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                
                # 转换为base64，使用JPEG格式和压缩
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG", quality=quality)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_size = len(img_str) / 1024  # 大小(KB)
                
                self.log(f"参考图像 {i+1} 编码大小: {img_size:.2f} KB")
                base64_images.append(img_str)
            
            return base64_images
        except Exception as e:
            self.log(f"图像转base64编码失败: {str(e)}")
            return None
    
    def generate_image(self, prompt, api_key, api_url, images, masks, seed, size="1024x1536"):
        """生成图像 - 使用gpt_edit_image函数处理图像编辑请求"""
        # 重置日志消息
        self.log_messages = []
        
        try:
            # 直接使用节点传入的种子值，ComfyUI已经处理了随机种子生成
            self.log(f"使用种子值: {seed}")
            
            # 设置随机种子，确保潜在的随机行为是可重现的
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # 获取API密钥和配置
            config = self.load_config()
            
            # 判断是否需要保存配置（API key、URL或model中任何一个与已保存的不同）
            should_save_config = False
            has_valid_api_key = api_key and len(api_key) > 10
            
            # 使用API密钥（优先使用用户输入，否则使用保存的）
            if has_valid_api_key:
                actual_api_key = api_key
                # 如果API密钥变化，需要保存配置
                if actual_api_key != config["api_key"]:
                    should_save_config = True
            else:
                actual_api_key = config["api_key"]
            
            # 使用API URL（检查是否有变化，避免空值覆盖已保存的值）
            if api_url and api_url.strip():  # 确保不是空字符串
                actual_api_url = api_url
                if actual_api_url != config["api_url"]:
                    should_save_config = True
            else:
                actual_api_url = config["api_url"]
            
            # 使用模型（检查是否有变化，避免空值覆盖已保存的值）
            # 如果需要保存并且有有效的API密钥，保存所有配置
            if should_save_config and actual_api_key:
                self.log("检测到配置变化，正在保存新配置...")
                self.save_config(actual_api_key, actual_api_url )
                
            # 记录使用的配置
            if actual_api_key:
                if should_save_config:
                    self.log("使用并保存了新的API配置")
                else:
                    self.log("使用现有API配置")
                
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的GPT API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                return (self.generate_empty_image(512, 512), full_text) # 返回空白图像
            

            self.log(f"api_url: {actual_api_url}")
            # 处理输入图像
            if images is None or images.shape[0] == 0:
                error_message = "错误: 未提供输入图像。"
                self.log(error_message)
                return (self.generate_empty_image(512, 512), error_message)
            
            # 获取第一张图像并保存为临时文件
            image_paths = self.encode_images_to_base64(images)
            image_path = image_paths[0]
            mask_path = None
            if masks is not None and masks.shape[0] > 0:
                mask_paths = self.encode_images_to_base64(masks)
                mask_path = mask_paths[0]
            else:
                mask_path = None

            try:
                result = gpt_edit_image(
                    image_path=image_path,
                    prompt=prompt,
                    api_key=actual_api_key,
                    endpoint=actual_api_url,
                    max_retry=3,
                    retry_interval=5,
                    mask_path=mask_path
                )
                
                self.log(f"gpt_edit_image处理完成，结果: {result}")
                
                if isinstance(result, str):  # 返回了base64编码的图像
                    # 解码base64图像
                    img_data = base64.b64decode(result)
                    buffer = BytesIO(img_data)
                    output_pil_image = Image.open(buffer)
                    
                    if output_pil_image.mode != 'RGB':
                        output_pil_image = output_pil_image.convert('RGB')
                    
                    # 转换为ComfyUI格式
                    output_array = np.array(output_pil_image).astype(np.float32) / 255.0
                    output_tensor = torch.from_numpy(output_array).unsqueeze(0)
                    
                    # 构建返回文本
                    result_text = f"## 图像编辑成功\n\n提示词: {prompt}\n"
                    result_text += f"\n输出图像: {output_pil_image.width}x{output_pil_image.height}"
                    result_text += f"\n种子: {seed}"
                    result_text += f"\n\n## 处理日志\n" + "\n".join(self.log_messages)
                    
                    return (output_tensor, result_text)
                
                else:
                    # 处理意外的返回值
                    error_message = f"错误: gpt_edit_image返回了意外的结果类型: {type(result)}"
                    self.log(error_message)
                    return (self.generate_empty_image(512, 512), error_message)
                
            except Exception as e:
                error_message = f"调用gpt_edit_image时出错: {str(e)}"
                self.log(error_message)
                traceback.print_exc()
                
                # 合并日志和错误信息
                full_text = f"## 错误\n" + error_message + f"\n\n## 请求信息\n提示词: {prompt}\n种子: {seed}\n\n## 处理日志\n" + "\n".join(self.log_messages)
                return (self.generate_empty_image(512, 512), full_text)
            
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"处理错误: {str(e)}")
            traceback.print_exc()
            
            # 合并日志和错误信息
            full_text = f"## 错误\n" + error_message + f"\n\n## 请求信息\n提示词: {prompt}\n种子: {seed}\n\n## 处理日志\n" + "\n".join(self.log_messages)
            return (self.generate_empty_image(512, 512), full_text)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "GPT-ImageGenerator": GPTImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT-ImageGenerator": "GPT4o Image Generation"
} 