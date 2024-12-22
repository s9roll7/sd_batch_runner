# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

import re
import logging
import os

import cv2
import numpy as np
from pathlib import Path
import torch
import timm
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sd_batch_runner.util import get_image_file_list


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



def prepare_wd14tagger():
    import os
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    os.makedirs("data/models/WD14tagger", exist_ok=True)
    for hub_file in [
        "selected_tags.csv",
    ]:
        path = Path(hub_file)

        saved_path = "data/models/WD14tagger" / path

        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="SmilingWolf/wd-eva02-large-tagger-v3", subfolder=PurePosixPath(path.parent), filename=PurePosixPath(path.name), local_dir="data/models/WD14tagger"
        )

def make_square(img, target_size=None):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    if target_size:
        desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im

def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class Tagger:
    def __init__(self, general_threshold, character_threshold, with_confidence, is_danbooru_format):
        prepare_wd14tagger()
        
        self.model = timm.create_model('hf-hub:SmilingWolf/wd-eva02-large-tagger-v3', pretrained=False).eval()
        state_dict = timm.models.load_state_dict_from_hf("SmilingWolf/wd-eva02-large-tagger-v3")
        self.model.load_state_dict(state_dict)

        data_config = timm.data.resolve_data_config(self.model.pretrained_cfg, model=self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        df = pd.read_csv("data/models/WD14tagger/selected_tags.csv")
        self.tag_names = df["name"].tolist()
        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.with_confidence = with_confidence
        self.is_danbooru_format = is_danbooru_format

    def __call__(
            self,
            image: Image,
            ):

        # Alpha to white
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]
        image = make_square(image)

        image = self.transform( Image.fromarray(image) ).unsqueeze(0)
        
        self.model = self.model.to("cuda")
        image = image.to("cuda")

        probs = self.model.forward(image)
        probs = torch.nn.functional.sigmoid(probs)

        image = image.to("cpu")
        probs = probs.to("cpu")

        labels = list(zip(self.tag_names, probs.squeeze(0).numpy()))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > self.general_threshold]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > self.character_threshold]
        character_res = dict(character_res)

        #logger.info(f"{rating=}")
        #logger.info(f"{general_res=}")
        #logger.info(f"{character_res=}")

        #general_res = {k:general_res[k] for k in (general_res.keys() - set(self.ignore_tokens)) }
        #character_res = {k:character_res[k] for k in (character_res.keys() - set(self.ignore_tokens)) }

        prompt = ""

        if self.with_confidence:
            prompt = [ f"({i}:{character_res[i]:.2f})" for i in (character_res.keys()) ]
            prompt += [ f"({i}:{general_res[i]:.2f})" for i in (general_res.keys()) ]
        else:
            prompt = [ i for i in (character_res.keys()) ]
            prompt += [ i for i in (general_res.keys()) ]

        prompt = ",".join(prompt)

        if not self.is_danbooru_format:
            prompt = prompt.replace("_", " ")

        #logger.info(f"{prompt=}")
        return prompt
    
    def __del__(self):
        if self.model:
            self.model = self.model.to("cpu")


def get_labels(frame_dir, general_threshold, character_threshold, with_confidence, is_danbooru_format):

    result = {}
    if os.path.isdir(frame_dir):
        
        png_list = get_image_file_list(frame_dir)

        with torch.no_grad():
            tagger = Tagger(general_threshold, character_threshold, with_confidence, is_danbooru_format)

            for p in tqdm( png_list, desc=f"WD14tagger"):
                result[p] = tagger(
                    image= Image.open(p)
                )

            tagger = None

        torch.cuda.empty_cache()

    return result

