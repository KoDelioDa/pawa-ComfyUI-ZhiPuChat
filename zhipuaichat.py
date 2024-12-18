import base64
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from zhipuai import ZhipuAI
from concurrent.futures import ThreadPoolExecutor


class ZhiPuChat:
    def __init__(self):
        # 用于暂存每张图片的 Base64 编码
        self.image_cache = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "please input your api_key"
                }),
                "model": (['glm-4v-plus', 'glm-4v', 'glm-4v-flash', "glm-4-plus", "glm-4-0520", "glm-4", "glm-4-air", "glm-4-airx", "glm-4-long", "glm-4-flashx", "glm-4-flash"], {'default': 'glm-4v-plus'}),
                "question": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "video_path": ("STRING", {
                    "multiline": False,
                    "default": "please input video path"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "type": (['video', 'image', 'text'], {'default': 'text'}),
            },
            "optional": {
                "images": ("IMAGE",),  # 单张图片输入
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)

    FUNCTION = "zhipuchat"

    CATEGORY = "ZhiPuChat"

    def tensor_to_base64(self, single_image):
        """将 Tensor 图像转换为 Base64 格式"""
        # 去掉 batch 维度，变为 [height, width, channels]
        single_image = single_image.squeeze(0)  # 去掉 batch 维度

        # 调整维度为 [channels, height, width]，适配 ToPILImage
        single_image = single_image.permute(2, 0, 1)  # [channels, height, width]

        # 转换为 PIL 图像
        transform = transforms.ToPILImage()
        pil_image = transform(single_image)

        # 使用 BytesIO 将图像保存为 PNG 格式到内存中
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")  # 保存为 PNG 格式
        buffer.seek(0)  # 将指针移到文件开头

        # 读取字节数据并转换为 base64
        image_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(image_bytes).decode()

        return img_base64

    def zhipuchat(self, question, api_key, model, seed, type, images=None, video_path=None):
        allowed_models = [
            "glm-4-plus",
            "glm-4-0520",
            "glm-4",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-long",
            "glm-4-flashx",
            "glm-4-flash"
        ]

        if api_key == "":
            return ("你没有填写api_key",)

        if type == "image":
            if images is not None:
                # 将 tensor 转为 base64 并暂存到缓存
                base64_image = self.tensor_to_base64(images)
                self.image_cache.append(base64_image)

            if model in allowed_models:
                return ("该模型不建议使用image，请更换模式",)

            # 如果所有图片已经完成，执行并发推理
            if len(self.image_cache) > 0:
                if question == "":
                    question = "请你分析图片内容"

                # 并发发送 base64 图片进行推理
                with ThreadPoolExecutor() as executor:
                    # 并发处理每张图片的推理
                    futures = [
                        executor.submit(self.send_image_request, img_base64, question, api_key, model, seed)
                        for img_base64 in self.image_cache
                    ]

                    # 收集每张图片的推理结果
                    responses = [future.result() for future in futures]

                # 清空缓存，避免重复处理
                self.image_cache.clear()

                # 返回所有结果
                return (" | ".join(responses),)

            return ("暂存图片处理中，请稍后再尝试",)

        if type == "video":
            if model == "glm-4v-plus":
                if question == "":
                    question = "请分析视频内容，并根据内容推断视频出现的声音，然后把推断的声音用文字直接描述出来，描述用tag提示词的方式，除此之外不要有任何多余的回复，不要有视频展示了xxx，可能xxx等多余的回复，只需要直接描述出声音"
                with open(video_path, 'rb') as video_file:
                    video_base = base64.b64encode(video_file.read()).decode('utf-8')
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": video_base
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ]
                # 调用模型 API
                client = ZhipuAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    request_id=seed
                )
                if response:
                    return (response.choices[0].message.content,)
            else:
                return ("该模型不建议使用video，请更换模式",)

        if type == "text":
            if question == "":
                return ("请输入问题",)
            if model in allowed_models:
                # 构造消息
                messages = [{"role": "user", "content": question}]
                
                # 调用模型 API
                client = ZhipuAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    request_id=seed
                )
                
                # 返回推理结果
                if response:
                    return (response.choices[0].message.content,)
                else:
                    return ("模型推理失败",)
            else:
                return ("该模型不支持 text 类型推理，请更换模式",)

        return ("类型不支持或未实现",)

    def send_image_request(self, img_base64, question, api_key, model, seed):
        """向模型发送单张图片的推理请求"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        # 调用模型 API
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            request_id=seed
        )
        # 返回推理结果
        if response:
            return response.choices[0].message.content
        return "推理失败"

NODE_CLASS_MAPPINGS = {
    "ZhiPuChat": ZhiPuChat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhiPuChat": "ZhiPuChatNode"
}
