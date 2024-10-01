import json
import logging
import time
from pathlib import Path
from datetime import datetime
import shutil
import random
import re

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


#######################################################################
def config_clear_cache():
    global _conf,_cn_conf,_lora_conf,_file_list_cache,_preset_tags_conf
    _conf = {}
    _cn_conf = {}
    _lora_conf = {}
    _file_list_cache = {}
    _preset_tags_conf = {}



#######################################################################
## config

CONFIG_FILE_PATH = "config.json"
_conf = {}

def get_config_dict():
    global _conf
    if not _conf:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            _conf = json.load(f)

    return _conf

def update_config():
    if _conf:
        json_text = json.dumps(_conf, indent=4, ensure_ascii=False)
        Path(CONFIG_FILE_PATH).write_text(json_text, encoding="utf-8")

def config_get_default_checkpoint():
    c = get_config_dict()
    return c["default_checkpoint"]
def config_set_default_checkpoint(new_val):
    c = get_config_dict()
    c["default_checkpoint"] = new_val
    update_config()


def config_set_current_lora_dir_env(new_val):
    c = get_config_dict()
    c["lora_dir_env"] = new_val
    update_config()

def config_get_current_lora_dir_env():
    c = get_config_dict()
    return c["lora_dir_env"]

def config_get_lora_dir_env_root_path():
    c = get_config_dict()
    p = Path("lora_dir_env") / Path(c["lora_dir_env"])
    p.mkdir(parents=True, exist_ok=True)
    return p



def config_get_default_generation_setting(is_txt2img):
    c = get_config_dict()
    if is_txt2img:
        return dict(**c["generation_setting_common"], **c["generation_setting_txt2img"])
    else:
        return dict(**c["generation_setting_common"], **c["generation_setting_img2img"])



def config_get_lora_generate_tag_enable_character():
    c = get_config_dict()
    return c["lora_generate_tag"]["enable_character"]

def config_get_lora_generate_tag_enable_style():
    c = get_config_dict()
    return c["lora_generate_tag"]["enable_style"]

def config_get_lora_generate_tag_enable_pose():
    c = get_config_dict()
    return c["lora_generate_tag"]["enable_pose"]

def config_get_lora_generate_tag_enable_item():
    c = get_config_dict()
    return c["lora_generate_tag"]["enable_item"]


def config_get_lora_generate_tag_th_character():
    c = get_config_dict()
    return c["lora_generate_tag"]["tag_th_character"]

def config_get_lora_generate_tag_th_style():
    c = get_config_dict()
    return c["lora_generate_tag"]["tag_th_style"]

def config_get_lora_generate_tag_th_pose():
    c = get_config_dict()
    return c["lora_generate_tag"]["tag_th_pose"]

def config_get_lora_generate_tag_th_item():
    c = get_config_dict()
    return c["lora_generate_tag"]["tag_th_item"]

def config_get_lora_generate_tag_prohibited_tags_character():
    c = get_config_dict()
    return c["lora_generate_tag"]["prohibited_tags_character"]

def config_get_lora_generate_tag_prohibited_tags_style():
    c = get_config_dict()
    return c["lora_generate_tag"]["prohibited_tags_style"]

def config_get_lora_generate_tag_prohibited_tags_pose():
    c = get_config_dict()
    return c["lora_generate_tag"]["prohibited_tags_pose"]

def config_get_lora_generate_tag_prohibited_tags_item():
    c = get_config_dict()
    return c["lora_generate_tag"]["prohibited_tags_item"]


def config_get_lbw_enable_character(lora_index):
    c = get_config_dict()
    lora_type = "character" if lora_index==0 else "character2"
    return c["lora_block_weight"][lora_type]["enable_lbw"]
def config_get_lbw_preset_character(lora_index):
    c = get_config_dict()
    lora_type = "character" if lora_index==0 else "character2"
    return c["lora_block_weight"][lora_type]["preset"]
def config_get_lbw_start_stop_step_character(lora_index):
    c = get_config_dict()
    lora_type = "character" if lora_index==0 else "character2"
    return c["lora_block_weight"][lora_type]["start_stop_step"]
def config_get_lbw_start_stop_step_value_character(lora_index):
    c = get_config_dict()
    lora_type = "character" if lora_index==0 else "character2"
    return c["lora_block_weight"][lora_type]["start_stop_step_value"]

def config_get_lbw_enable_style(lora_index):
    c = get_config_dict()
    lora_type = "style" if lora_index==0 else "style2"
    return c["lora_block_weight"][lora_type]["enable_lbw"]
def config_get_lbw_preset_style(lora_index):
    c = get_config_dict()
    lora_type = "style" if lora_index==0 else "style2"
    return c["lora_block_weight"][lora_type]["preset"]
def config_get_lbw_start_stop_step_style(lora_index):
    c = get_config_dict()
    lora_type = "style" if lora_index==0 else "style2"
    return c["lora_block_weight"][lora_type]["start_stop_step"]
def config_get_lbw_start_stop_step_value_style(lora_index):
    c = get_config_dict()
    lora_type = "style" if lora_index==0 else "style2"
    return c["lora_block_weight"][lora_type]["start_stop_step_value"]

def config_get_lbw_enable_pose(lora_index):
    c = get_config_dict()
    lora_type = "pose" if lora_index==0 else "pose2"
    return c["lora_block_weight"][lora_type]["enable_lbw"]
def config_get_lbw_preset_pose(lora_index):
    c = get_config_dict()
    lora_type = "pose" if lora_index==0 else "pose2"
    return c["lora_block_weight"][lora_type]["preset"]
def config_get_lbw_start_stop_step_pose(lora_index):
    c = get_config_dict()
    lora_type = "pose" if lora_index==0 else "pose2"
    return c["lora_block_weight"][lora_type]["start_stop_step"]
def config_get_lbw_start_stop_step_value_pose(lora_index):
    c = get_config_dict()
    lora_type = "pose" if lora_index==0 else "pose2"
    return c["lora_block_weight"][lora_type]["start_stop_step_value"]

def config_get_lbw_enable_item(lora_index):
    c = get_config_dict()
    lora_type = "item" if lora_index==0 else "item2"
    return c["lora_block_weight"][lora_type]["enable_lbw"]
def config_get_lbw_preset_item(lora_index):
    c = get_config_dict()
    lora_type = "item" if lora_index==0 else "item2"
    return c["lora_block_weight"][lora_type]["preset"]
def config_get_lbw_start_stop_step_item(lora_index):
    c = get_config_dict()
    lora_type = "item" if lora_index==0 else "item2"
    return c["lora_block_weight"][lora_type]["start_stop_step"]
def config_get_lbw_start_stop_step_value_item(lora_index):
    c = get_config_dict()
    lora_type = "item" if lora_index==0 else "item2"
    return c["lora_block_weight"][lora_type]["start_stop_step_value"]



def config_get_adetailer_setting():
    c = get_config_dict()
    return c["adetailer"]

def config_get_default_prompt_gen_setting():
    c = get_config_dict()
    return c["prompt_gen_setting"]

def config_get_default_overwrite_generation_setting():
    c = get_config_dict()
    return c["overwrite_generation_setting"]

def config_get_segment_anything_sam_model_name():
    c = get_config_dict()
    return c["segment_anything"]["sam_model_name"]

def config_get_segment_anything_dino_model_name():
    c = get_config_dict()
    return c["segment_anything"]["dino_model_name"]




#######################################################################
## controlnet

CONTROLNET_FILE_PATH = "controlnet.json"
_cn_conf = {}

def get_cn_config_dict():
    global _cn_conf
    if not _cn_conf:
        with open(CONTROLNET_FILE_PATH, "r", encoding="utf-8") as f:
            _cn_conf = json.load(f)

    return _cn_conf

def get_controlnet_setting(name):
    c = get_cn_config_dict()
    return c[name]

#######################################################################
## lora dir

LORA_DIR_FILE_PATH = "lora_dir.json"
_lora_conf = {}

def get_lora_config_dict():
    global _lora_conf
    if not _lora_conf:
        with open(LORA_DIR_FILE_PATH, "r", encoding="utf-8") as f:
            _lora_conf = json.load(f)

    return _lora_conf

def config_get_item_lora_dir_path():
    c = get_lora_config_dict()
    env = config_get_current_lora_dir_env()
    return Path(c[env]["item_dir_path"])

def config_get_pose_lora_dir_path():
    c = get_lora_config_dict()
    env = config_get_current_lora_dir_env()
    return Path(c[env]["pose_dir_path"])

def config_get_style_lora_dir_path():
    c = get_lora_config_dict()
    env = config_get_current_lora_dir_env()
    return Path(c[env]["style_dir_path"])

def config_get_character_lora_dir_path():
    c = get_lora_config_dict()
    env = config_get_current_lora_dir_env()
    return Path(c[env]["character_dir_path"])

def config_get_lora_dir_env_list():
    c = get_lora_config_dict()
    logger.info(f"{c=}")
    return list(c.keys())



#######################################################################
## preset tags

PRESET_TAGS_FILE_PATH = "preset_tags.json"
_preset_tags_conf = {}

def get_preset_tags_config_dict():
    global _preset_tags_conf
    if not _preset_tags_conf:
        with open(PRESET_TAGS_FILE_PATH, "r", encoding="utf-8") as f:
            _preset_tags_conf = json.load(f)

    return _preset_tags_conf

def config_get_preset_tags_info(preset_name):
    c = get_preset_tags_config_dict()
    return c[preset_name]



#######################################################################


def get_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_file_list_cache={}

def select_one_file(dir_path:Path, suffixes, is_random=True):

    key = dir_path
    if key not in _file_list_cache:
        file_list = [p for p in dir_path.glob("**/*")]
        file_list = [p for p in file_list if p.suffix in suffixes]
        if len(file_list) == 0:
            raise ValueError(f"file not found in {dir_path=}")
        if is_random:
            random.shuffle(file_list)
        _file_list_cache[key] = file_list

    item = _file_list_cache[key].pop(0)
    if len(_file_list_cache[key]) == 0:
        _file_list_cache.pop(key)
    return item


_video_cache={}

def select_frame(movie_path:Path, interval_sec:float):
    import av
    key = movie_path
    if key not in _video_cache:
        video = av.open( movie_path )
        _video_cache[key] = [video, 0]
    
    video = _video_cache[key][0]
    cur_sec = _video_cache[key][1]

    stream = video.streams.video[0]
    offset = int(cur_sec // stream.time_base)
    logger.info(f"{offset=}")

    video.seek(
        offset = offset,
        any_frame=False,
        backward=True,
        stream=stream
    )

    while True:
        frame = next(video.decode(video=0), None)
        if frame is None:
            raise ValueError(f"seek failed. {movie_path=} {interval_sec=} {cur_sec=}")

        logger.debug(f"seeking :{frame.time}")
        if frame.time >= cur_sec:
            break
    
    image = Image.fromarray( frame.to_ndarray(format="rgb24") )

    cur_sec += interval_sec
    if float(stream.duration * stream.time_base) <= cur_sec:
        cur_sec = 0
    _video_cache[key][1] = cur_sec

    return image


def clear_video_cache():
    for key in list(_video_cache.keys()):
        item = _video_cache.pop(key)
        item[0].close()
    







#########################################################
# from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/infotext_utils.py

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")

def parse_generation_parameters(x: str, skip_fields: list[str] | None = None):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """

    def unquote(text):
        if len(text) == 0 or text[0] != '"' or text[-1] != '"':
            return text

        try:
            return json.loads(text)
        except Exception:
            return text


    if skip_fields is None:
        skip_fields = []

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    for k, v in re_param.findall(lastline):
        try:
            if v[0] == '"' and v[-1] == '"':
                v = unquote(v)

            m = re_imagesize.match(v)
            if m is not None:
                res[f"{k}-1"] = m.group(1)
                res[f"{k}-2"] = m.group(2)
            else:
                res[k] = v
        except Exception:
            print(f"Error parsing \"{k}: {v}\"")

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires schedule type" not in res:
        res["Hires schedule type"] = "Use same scheduler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    if "Mask mode" not in res:
        res["Mask mode"] = "Inpaint masked"

    if "Masked content" not in res:
        res["Masked content"] = 'original'

    if "Inpaint area" not in res:
        res["Inpaint area"] = "Whole picture"

    if "Masked area padding" not in res:
        res["Masked area padding"] = 32


    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    if "VAE Encoder" not in res:
        res["VAE Encoder"] = "Full"

    if "VAE Decoder" not in res:
        res["VAE Decoder"] = "Full"

    if "FP8 weight" not in res:
        res["FP8 weight"] = "Disable"

    if "Cache FP16 weight for LoRA" not in res and res["FP8 weight"] != "Disable":
        res["Cache FP16 weight for LoRA"] = False

    if "Refiner switch by sampling steps" not in res:
        res["Refiner switch by sampling steps"] = False

    for key in skip_fields:
        res.pop(key, None)

    return res



####################################################################
def crop_mask(im:Image.Image, th):
    im_array = np.array(im)
    coords = np.argwhere(im_array > th)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    cropped = im_array[x_min:x_max+1, y_min:y_max+1]
    return Image.fromarray(cropped)

def get_box_of_mask(im:Image.Image, th):
    im_array = np.array(im)
    coords = np.argwhere(im_array > th)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return (x_min, x_max+1), (y_min, y_max+1)

def get_center_of_mask(im:Image.Image, th):
    (x0,x1),(y0,y1) = get_box_of_mask(im,th)
    return ((x0+x1)//2 , (y0+y1)//2)

def create_focus_image(im:Image.Image, pos, scale):
    w, h = im.size
    logger.info(f"{scale=}")

    scale = float(scale)

    im_array = np.array(im)

    if pos:
        cx = pos[1]
        cy = pos[0]
    else:
        cx = w/2
        cy = h/2

    logger.info(f"{scale=}")
    if scale > 1.0:
        logger.info(f"scale > 1.0")
        cxmin = (w/scale)/2
        cxmax = w- (w/scale)/2

        logger.info(f"cxmin:{cxmin} cxmax:{cxmax}")

        cx = min(max(cx, cxmin), cxmax)

        cymin = (h/scale)/2
        cymax = h- (h/scale)/2

        logger.info(f"cymin:{cymin} cymax:{cymax}")

        cy = min(max(cy, cymin), cymax)
    

    logger.info(f"x:{pos[1]} y:{pos[0]}")
    logger.info(f"focus to {cx=} {cy=}")

    M = cv2.getRotationMatrix2D((float(cx), float(cy)), 0, float(scale))
    expand_im = cv2.warpAffine(im_array, M, (w, h))

    return Image.fromarray(expand_im)




####################################################################

_thumb_cache = {}

def get_thumb(path:Path, size):
    global _thumb_cache

    def im_2_b64(image):
        from io import BytesIO
        import base64
        buff = BytesIO()
        image.convert('RGB').save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue())
        return img_str

    key = (path,size)
    if key not in _thumb_cache:
        thumb = Image.open(path)
        thumb.thumbnail(size=size)
        _thumb_cache[key] = im_2_b64(thumb)
    
    return _thumb_cache[key].decode('ascii')


####################################################################

def config_restore_files_if_needed():
    config_files = [
        "config.json",
        "preset_tags.json",
        "lora_dir.json",
        "controlnet.json",
    ]

    for c in config_files:
        if Path(c).is_file() == False:
            shutil.copy( Path(f"default_config/default_{c}"), c)

