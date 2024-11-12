# LongVU

> **LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding**
>
> <a href='https://arxiv.org/abs/2410.17434'><img src='https://img.shields.io/badge/arXiv-paper-red'></a> <a href='https://vision-cair.github.io/LongVU'><img src='https://img.shields.io/badge/project-LongVU-blue'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/LongVU'><img src='https://img.shields.io/badge/demo-space-blue'></a> <a href='https://huggingface.co/collections/Vision-CAIR/longvu-67181d2debabfc1eb050c21d'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 

<div align="center">
    <a href='https://vision-cair.github.io/LongVU'><img src="https://longvu.s3.amazonaws.com/assets/demo.gif" alt="Demo GIF" style="width: 100%; max-width: 650px;"></a>
</div>

## :rocket: Quick Start

Try our model on [HF 🤗 Demo](https://huggingface.co/spaces/Vision-CAIR/LongVU)

Or deploy from local

### Linux

```
git clone https://github.com/Vision-CAIR/LongVU
cd LongVU
conda create -n longvu python=3.10
conda activate longvu
pip install -r requirements.txt
```

Download our checkpoints and put it under `./checkpoints`

| Modality | LongVU_Qwen2_7B | LongVU_Llama3_2_3B |
:--------------------------:| :--------------------------:|:--------------------------:
| Image | [Download](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B_img) | [Download](https://huggingface.co/Vision-CAIR/LongVU_Llama3_2_3B_img) |
| Video | [Download](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B) | [Download](https://huggingface.co/Vision-CAIR/LongVU_Llama3_2_3B) |

Run demo `python app.py` locally with minimum 40G GPU.

<details>
  <summary>Click for quick inference code</summary>
    
```python
import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "./checkpoints/longvu_qwen", None, "cambrian_qwen",
)

model.eval()
video_path = "./examples/video1.mp4"
qs = "Describe this video in detail"

vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = float(vr.get_avg_fps())
frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
video = []
for frame_index in frame_indices:
    img = vr[frame_index].asnumpy()
    video.append(img)
video = np.stack(video)
image_sizes = [video[0].shape[:2]]
video = process_images(video, image_processor, model.config)
video = [item.unsqueeze(0) for item in video]

qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates["qwen"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.2,
        max_new_tokens=128,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )
pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
```
    
</details>

### Windows

Thanks the detailed instruction [(here)](https://github.com/Vision-CAIR/LongVU/issues/6) from @ipeevski for developing on Windows system with 24GB VRAM.

## Training

### Dataset

+ image-text stage: [LLaVA-OneVision-Single](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)
+ video-text stage: [VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)

### Scripts

Experiments are run on 64 H100-96G

Download [image_json_file](https://huggingface.co/datasets/shenxq/OneVision/blob/main/onevision.json) and [video_json_file](https://huggingface.co/datasets/shenxq/VideoChat2/blob/main/train_video_data.json).

We also provide [row_video_data](https://huggingface.co/datasets/shenxq/VideoChat2) for easy downloading.

Modify the PATH_TO_JSON and PATH_TO_FOLDER arguments in the training scripts to your save folder.

```
PATH_TO_JSON=""
PATH_TO_FOLDER=""
```
Training your own model
```
# image sft
sh scripts/train_image_qwen.sh
sh scripts/train_image_llama3_2.sh
```

Modify PREV_STAGE_CHECKPOINT in the training scripts to your first stage model path

Change `image_token_len` and `query_num_list` in `config.json` to 144

```
# video sft
sh scripts/train_video_qwen.sh
sh scripts/train_video_llama3_2.sh
```

## Evaluation

See detailed evaluation code in [eval.md](https://github.com/Vision-CAIR/LongVU/blob/main/docs/eval.md)

## Acknowledgement

+ The model architecture of LongVU follows [LLaVA](https://github.com/haotian-liu/LLaVA) and [Cambrian](https://github.com/cambrian-mllm/cambrian)
+ We base [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) as the language backbone
+ We use [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [DINOv2](https://huggingface.co/facebook/dinov2-giant) as the vision encoder

## Citation

```
@article{shen2024longvu,
  author ={Shen, Xiaoqian and Xiong, Yunyang and Zhao, Changsheng and Wu, Lemeng and Chen, Jun and Zhu, Chenchen and Liu, Zechun and Xiao, Fanyi and Varadarajan, Balakrishnan and Bordes, Florian and Liu, Zhuang and Xu, Hu and J. Kim, Hyunwoo and Soran, Bilge and Krishnamoorthi, Raghuraman and Elhoseiny, Mohamed and Chandra, Vikas},
  title = {LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding},
  journal = {arXiv preprint arXiv:2410.17434},
  year = {2024},
}
```
