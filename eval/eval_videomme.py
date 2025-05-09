# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.


import datetime
import json
import logging
import os
import re
import shutil
import uuid
from itertools import chain
import argparse

import sys
sys.path.append('./')
import numpy as np
import pysubs2

import torch

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord
from pyarrow import parquet as pq
from torch import distributed as dist
from tqdm import tqdm

from transformers.trainer_pt_utils import IterableDatasetShard

# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)

    # Convert PyArrow Table to pandas DataFrame
    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():

        if len(jsons) < int(record.video_id):
            jsons.append(
                {
                    "video_id": record.video_id,
                    "youtube_id": record.videoID,
                    "url": record.url,
                    "duration": record.duration,
                    "domain": record.domain,
                    "sub_category": record.sub_category,
                    "questions": [
                        {
                            "question_id": record.question_id,
                            "task_type": record.task_type,
                            "question": record.question,
                            "choices": list(record.options),
                            "answer": record.answer,
                        }
                    ],
                }
            )
        else:
            jsons[-1]["questions"].append(
                {
                    "question_id": record.question_id,
                    "task_type": record.task_type,
                    "question": record.question,
                    "choices": list(record.options),
                    "answer": record.answer,
                }
            )

    return jsons


class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        data_path: str,
    ) -> None:
        super(EvalDataset, self).__init__()
        logging.warning("Loading data...")

        self.data_path = data_path

        data_list = load_parquet(
            os.path.join(self.data_path, "test-00000-of-00001.parquet")
        )

        list_data_dict = []

        for item in data_list:
            video_ytid = item["url"].split("watch?v=")[-1]
            video_path = os.path.join(self.data_path, "data", f"{video_ytid}.mp4")
            for fmt in self.video_formats:  # Added this line
                temp_path = os.path.join(self.data_path, "data", f"{video_ytid}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            subtitle_path = os.path.join(
                self.data_path, "subtitle", f"{video_ytid}.srt"
            )

            list_data_dict.append(
                {
                    "questions": item["questions"],
                    "video": video_path,
                    "subtitle": subtitle_path,
                    "video_name": video_ytid,
                    "duration": item["duration"],
                }
            )

        # pyre-fixme[4]: Attribute must be annotated.
        self.data = list_data_dict

    def __len__(self) -> int:
        return len(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        return iter(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getitem__(self, i):
        return self.data[i]


def train(args) -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    
    version = args.version
    model_name = args.model_name
    model_path = args.model_path

    torch.distributed.barrier()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,  # pyre-fixme
        None,
        model_name,
        device_map=None,
    )
    model.get_model().config.drop_threshold = 0.65
    model.config.use_cache = True
    model.cuda()
    dataset = EvalDataset(
        data_path=args.data_path,
    )
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()
    output = []
    final_output = [None] * world_size

    for line in tqdm(shard_dataset):
        video_name = line["video_name"]
        video_path = line["video"]
        subtitle_path = line["subtitle"]
        questions = line["questions"]

        if os.path.exists(video_path):
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = round(vr.get_avg_fps())
            sample_fps = 0.5
            frame_idx = [i for i in range(0, len(vr), round(fps / sample_fps))]
            video = vr.get_batch(frame_idx).asnumpy()
            
            image_sizes = [video[0].shape[:2]]
            video = process_images(video, image_processor, model.config)
            video = [item.unsqueeze(0) for item in video]
        else:
            frame_idx = [1]
            fps = 1
            print(f"video_path {video_path} does not exist")

            video = np.zeros((1, 1024, 1024, 3)).astype(np.uint8)
            image_sizes = [(1024, 1024)]
            video = process_images(video, image_processor, model.config)

        if (
            os.path.exists(video_path)
            and os.path.exists(subtitle_path)
        ):
            subs = pysubs2.load(subtitle_path, encoding="utf-8")
            subtitles = []
            for select_id in range(0, len(frame_idx)):
                seleced_frame_id = frame_idx[select_id]
                sub_text = ""
                cur_time = pysubs2.make_time(fps=fps, frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        break
                if sub_text.strip():
                    if (
                        "[Music]" not in sub_text
                        and "[Applause]" not in sub_text
                        and sub_text not in subtitles
                    ):
                        if len(subtitles) > 0:
                            if sub_text not in subtitles[-1]:
                                subtitles.append(sub_text)
                        else:
                            subtitles.append(sub_text)
            if len(tokenizer("\n".join(subtitles)).input_ids) > 6000:
                interval = len(subtitles) // 200
                indices = np.arange(0, len(subtitles), interval)
                subtitles = [subtitles[i] for i in indices]
            subtitles = "\n".join(subtitles)
            subtitles = f"This video's subtitles are listed below:\n{subtitles}\n"
        else:
            subtitles = ""

        for question in questions:
            q = question["question"]
            ops = question["choices"]
            instruct = f"Question: {q}\n"
            instruct += "Options:\n"
            for op in ops:
                instruct += f"{op}\n"
            instruct += (
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            )
            qs = subtitles + instruct

            if getattr(model.config, "mm_use_im_start_end", False):
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[version].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            if "llama3" in version:
                input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=5,  
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            if isinstance(output_ids, tuple):
                output_ids = output_ids[0]
            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            if pred.endswith(stop_str):
                pred = pred[: -len(stop_str)]
                pred = pred.strip()
            pred = pred.replace("Answer", "")

            letters = ["A", "B", "C", "D"]

            pred_answer = re.findall("[\(\ \[]*([A-D])[\)\.\ \]]*", pred)
            if not pred_answer: #If list is empty
                pred_idx = 2
                pred = letters[pred_idx]
            else:
                pred_answer = pred_answer[0].strip()
                pred_answer = pred_answer.strip("()")
                if pred_answer in letters:
                    pred_idx = letters.index(pred_answer)
                    pred = letters[pred_idx]
                else:
                    print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
                    pred_idx = 2
                    pred = letters[pred_idx]

            ans_id = uuid.uuid4()
            output.append(
                {
                    "question": question["question"],
                    "answer": question["answer"],
                    "pred": pred,
                    "answer_id": str(ans_id),
                    "model_id": model_name,
                    "video_name": video_name,
                    "metadata": {},
                    "duration": line["duration"],
                }
            )

    dist.barrier()
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output))
    global_rank = dist.get_rank()
    if global_rank == 0:
        if os.path.exists("/tmp/generated_text"):
            shutil.rmtree("/tmp/generated_text")
        os.mkdir("/tmp/generated_text")

        with open(
            os.path.join("/tmp/generated_text", "outputs.json"),
            "w",
        ) as f:
            json.dump(all_output, f)

        correct = 0
        duration_correct = {"short": 0, "medium": 0, "long": 0}
        duration_all = {"short": 0, "medium": 0, "long": 0}

        for output in all_output:
            duration = output["duration"]
            duration_all[duration] += 1
            if output["pred"] == output["answer"]:
                correct += 1
                duration_correct[duration] += 1

        result = {"averge_acc": correct / len(all_output)}

        for duration, correct_count in duration_correct.items():
            result[f"{duration}_acc"] = (
                correct_count / duration_all[duration]
                if duration_all[duration] > 0
                else 0
            )

        print(f"Accuracy: {result}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./checkpoints/longvu_qwen")
    parser.add_argument('--model_name', default="cambrian_qwen")
    parser.add_argument('--version', default="qwen")
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    if "llama3" in args.version:
        args.model_name = "cambrian_llama3"

    train(args)
