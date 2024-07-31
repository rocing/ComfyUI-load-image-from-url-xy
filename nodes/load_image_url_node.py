import time

from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO
import os
import numpy as np


def pil2tensor(img):
    output_images = []
    output_masks = []
    try:
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def load_image(image_source, timeout=(5, 10)):
    if image_source.startswith('http'):
        print(image_source)
        max_retries = 3
        file_name = image_source.split('/')[-1]
        for attempt in range(max_retries):
            try:
                response = requests.get(image_source, stream=True, timeout=timeout)
                response.raise_for_status()
                original_size = int(response.headers.get('Content-Length', 0))
                print(original_size)
                img_data = BytesIO(response.content)
                img = Image.open(img_data)
                image_size = img_data.getbuffer().nbytes
                print(image_size)
                if image_size != original_size:
                    print(
                        f"下载的文件大小不一致，原始大小：{original_size}, 下载大小：{img_data.getbuffer().nbytes}，正在重试...")
                    continue
                return img, file_name

            except Exception as e:
                print(f"下载图片出错: {e}")
                if attempt < max_retries - 1:
                    print(f"正在进行第 {attempt + 1} 次重试...")
                    time.sleep(1)
                else:
                    print(f"图片下载失败，请检查网络连接和图片链接。")
                    return None, None
            # except IOError:
            #     print("图片数据错误，无法打开。")
            #     return None, None
    else:
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name


class LoadImageByUrlOrPathXY:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "connect_timeout": ("INT", {"default": 1}),
                "read_timeout": ("INT", {"default": 10})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load"
    CATEGORY = "image"

    def load(self, url_or_path, connect_timeout, read_timeout):
        print(url_or_path)
        img, name = load_image(url_or_path, (connect_timeout, read_timeout))
        img_out, mask_out = pil2tensor(img)
        return (img_out, mask_out)


if __name__ == "__main__":
    img, name = load_image("http://10.28.1.42:8018/api/inputView?filename=114a1c455d8090d8afa3cbc7f2c72fa6.png")
    img_out, mask_out = pil2tensor(img)
