import json
import logging
import time
from pathlib import Path
from datetime import datetime
import math
import random
import re
from enum import Enum


from sd_batch_runner.util import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


re_word = re.compile(r"[-_\w']+")



class LoraType(str, Enum):
    Item = "item"
    Pose = "pose"
    Style = "style"
    Character = "char"
    All = "all"


class Lora():

    static_instance_map={}

    def __init__(self, user_data_path:Path, data_path:Path, tag_th, prohibited_tags, enable_gen_tag):

        r = config_get_lora_dir_env_root_path()
        user_data_path = r / user_data_path
        data_path = r / data_path

        self.data={}
        if data_path.is_file():
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        self.user_data={}
        if user_data_path.is_file():
            with open(user_data_path, "r", encoding="utf-8") as f:
                self.user_data = json.load(f)

        self.default_weight = 1.0
        self.tag_th = tag_th
        self.prohibited_tags = prohibited_tags
        self.enable_gen_tag = enable_gen_tag

        self.random_order_cache = {}
    
    def _select(self, key):

        logger.debug(f"select {key=}")

        cur_item = self.data.get(key, {})

        if not cur_item:
            return None

        user_item = self.user_data.get(key, {})
        for k in user_item:
            cur_item[k] = user_item[k]
        
        w = cur_item.get("weight", self.default_weight)

        a = Path(key).stem
        b = w
        c = ""

        if user_item:
            trigger = user_item.get("trigger", [])
            if trigger:
                index = random.randrange(0, len(trigger))
                c = trigger[index]
        
        if not c:
            tags = cur_item.get("tags", [])
            if self.enable_gen_tag and tags:
                c = generate_prompt_from_tags(tags, self.tag_th, self.prohibited_tags)
            else:
                trigger = cur_item.get("trigger", [])
                if trigger:
                    index = random.randrange(0, len(trigger))
                    c = trigger[index]

        return [a,b,c]
    
    def _select_one(self, filter):
        item_list = self.get_file_list()
        if not item_list:
            return None
        
        if (filter not in self.random_order_cache) or (not self.random_order_cache[filter]):
            if filter != "":
                filter_list = filter.split("|")
                filter_list = [f for f in filter_list if f]
                
                result = []
                for f in filter_list:
                    result += [name for name in item_list if name.lower().find(f) != -1]
                
                item_list = list(dict.fromkeys(result))

                if len(item_list) == 0:
                    return None
            else:
                random.shuffle(item_list)
            self.random_order_cache[filter] = item_list
        

        key = self.random_order_cache[filter].pop(0)

        return self._select(key)
    
    def _clear_filter(self):
        self.random_order_cache.clear()
    
    def get_file_dir(self):
        raise NotImplementedError()
    
    def get_file_list(self):
        tmp = list(self.data.keys())
        tmp.sort()
        return tmp
    
    @classmethod
    def _create(cls, lora_type:LoraType):
        if lora_type not in Lora.static_instance_map:
            if lora_type == LoraType.Item:
                Lora.static_instance_map[lora_type] = ItemLora()
            elif lora_type == LoraType.Pose:
                Lora.static_instance_map[lora_type] = PoseLora()
            elif lora_type == LoraType.Style:
                Lora.static_instance_map[lora_type] = StyleLora()
            elif lora_type == LoraType.Character:
                Lora.static_instance_map[lora_type] = CharacterLora()

    @classmethod
    def select(cls, lora_type:LoraType, key):
        if lora_type == LoraType.All:
            types = [LoraType.Item,LoraType.Pose,LoraType.Style,LoraType.Character]
        else:
            types = [lora_type]
        
        for t in types:
            Lora._create(t)

        result = None
        for t in types:
            result = Lora.static_instance_map[t]._select(key)
            if result != None:
                break
        
        return result

    @classmethod
    def select_one(cls, lora_type:LoraType, filter=""):
        if lora_type == LoraType.All:
            types = [LoraType.Item,LoraType.Pose,LoraType.Style,LoraType.Character]
        else:
            types = [lora_type]
        
        for t in types:
            Lora._create(t)

        result = None
        for t in types:
            result = Lora.static_instance_map[t]._select_one(filter)
            if result != None:
                break
        
        return result

    @classmethod
    def create_instance(cls, lora_type:LoraType):
        if lora_type == LoraType.Item:
            return ItemLora()
        elif lora_type == LoraType.Pose:
            return PoseLora()
        elif lora_type == LoraType.Style:
            return StyleLora()
        elif lora_type == LoraType.Character:
            return CharacterLora()
        else:
            raise NotImplementedError()

def lora_clear_cache():
    Lora.static_instance_map.clear()


class ItemLora(Lora):
    def __init__(self):
        tag_th = config_get_lora_generate_tag_th_item()
        prohibited_tags = config_get_lora_generate_tag_prohibited_tags_item()
        enable_gen_tag = config_get_lora_generate_tag_enable_item()
        super().__init__(Path("user_lora.json"), Path("item_lora.json"), tag_th, prohibited_tags, enable_gen_tag)
    def get_type(self):
        return LoraType.Item
    def get_file_dir(self):
        return config_get_item_lora_dir_path()

class PoseLora(Lora):
    def __init__(self):
        tag_th = config_get_lora_generate_tag_th_pose()
        prohibited_tags = config_get_lora_generate_tag_prohibited_tags_pose()
        enable_gen_tag = config_get_lora_generate_tag_enable_pose()
        super().__init__(Path("user_lora.json"), Path("pose_lora.json"), tag_th, prohibited_tags, enable_gen_tag)
    def get_type(self):
        return LoraType.Pose
    def get_file_dir(self):
        return config_get_pose_lora_dir_path()

class StyleLora(Lora):
    def __init__(self):
        tag_th = config_get_lora_generate_tag_th_style()
        prohibited_tags = config_get_lora_generate_tag_prohibited_tags_style()
        enable_gen_tag = config_get_lora_generate_tag_enable_style()
        super().__init__(Path("user_lora.json"), Path("style_lora.json"), tag_th, prohibited_tags, enable_gen_tag)
    def get_type(self):
        return LoraType.Style
    def get_file_dir(self):
        return config_get_style_lora_dir_path()

class CharacterLora(Lora):
    def __init__(self):
        tag_th = config_get_lora_generate_tag_th_character()
        prohibited_tags = config_get_lora_generate_tag_prohibited_tags_character()
        enable_gen_tag = config_get_lora_generate_tag_enable_character()
        super().__init__(Path("user_lora.json"), Path("character_lora.json"), tag_th, prohibited_tags, enable_gen_tag)
    def get_type(self):
        return LoraType.Character
    def get_file_dir(self):
        return config_get_character_lora_dir_path()


def generate_prompt_from_tags(tags, th = -1, prohibited_tags=[]):
    max_count = None
    res = []
    for tag, count in tags:
        if not max_count:
            max_count = count

        if tag in prohibited_tags:
            logger.debug(f"ignore {tag=}")
            continue

        if th < 0:
            v = random.random() * max_count
        else:
            v = th * max_count
        if count > v:
            for x in "({[]})":
                tag = tag.replace(x, '\\' + x)
            res.append(tag)

    return ", ".join(sorted(res))

# from https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/master/modules/sd_models.py
def get_train_tags_from_safetensors(file_path):
    
    def read_metadata_from_safetensors(filename):
        with open(filename, mode="rb") as file:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)

            assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"

            res = {}

            try:
                json_data = json_start + file.read(metadata_len-2)
                json_obj = json.loads(json_data)
                for k, v in json_obj.get("__metadata__", {}).items():
                    res[k] = v
                    if isinstance(v, str) and v[0:1] == '{':
                        try:
                            res[k] = json.loads(v)
                        except Exception:
                            pass
            except Exception:
                logger.error(f"Error reading metadata from file: {filename}")

            return res
    
    def build_tags(metadata):
        def is_non_comma_tagset(tags):
            average_tag_length = sum(len(x) for x in tags.keys()) / len(tags)
            return average_tag_length >= 16
        
        tags = {}

        ss_tag_frequency = metadata.get("ss_tag_frequency", {})
        if ss_tag_frequency is not None and hasattr(ss_tag_frequency, 'items'):
            for _, tags_dict in ss_tag_frequency.items():
                for tag, tag_count in tags_dict.items():
                    tag = tag.strip()
                    tags[tag] = tags.get(tag, 0) + int(tag_count)

        if tags and is_non_comma_tagset(tags):
            new_tags = {}

            for text, text_count in tags.items():
                for word in re.findall(re_word, text):
                    if len(word) < 3:
                        continue

                    new_tags[word] = new_tags.get(word, 0) + text_count

            tags = new_tags

        ordered_tags = sorted(tags.keys(), key=tags.get, reverse=True)

        return [(tag, tags[tag]) for tag in ordered_tags]
    
    #######################################

    metadata_tags_order = {"ss_sd_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

    metadata = read_metadata_from_safetensors(file_path)

    if metadata:
        m = {}
        for k, v in sorted(metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
            m[k] = v
        
        return build_tags(m)
    else:
        return []


def update_lora(lora_dir_path:Path, json_path:Path, is_overwrite):

    data = {}

    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    def search_weight(info, lora_name:str):
        imgs = info.get("images",[])
        for img in imgs:
            meta = img.get("meta",{})
            if meta:
                ress = meta.get("resources", [])
                for res in ress:
                    name = res.get("name", "")
                    if lora_name.startswith(name):
                        weight = res.get("weight", None)
                        if weight is not None:
                            return weight

    result = []

    safetensors_list = [ p for p in lora_dir_path.glob("**/*") if p.suffix in (".safetensors")]

    for p in safetensors_list:
        data_exist = p.name in data

        if is_overwrite == False:
            if data_exist:
                result.append((p.name , "skip"))
                continue
        
        trigger = []
        updated_date = ""
        name = ""

        info = {}
        if p.with_suffix(".civitai.info").is_file():
            with open(p.with_suffix(".civitai.info"), "r") as f:
                info = json.load(f)
            
            trigger = info.get("trainedWords", [])
            updated_date = info.get("updatedAt", "")
            name = info.get("name", "")
        

        tags = get_train_tags_from_safetensors(p)


        data[p.name] = {
            "name" : name,
            "updated_date" : updated_date,
            "trigger": trigger,
            "tags" : tags
        }

        w = search_weight(info, p.stem)
        if w is not None:
            data[p.name]["weight"] = w
        
        result.append((p.name , "update" if data_exist else "add"))
    
    safetensors_list = [ p.name for p in safetensors_list]
    key_list = list(data.keys())

    key_only = set(key_list) - set(safetensors_list)
    key_only = list(key_only)

    for k in key_only:
        data.pop(k)
        result.append((k , "delete"))


    
    json_text = json.dumps(data, indent=4, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")

    return result, len(data.keys())

def print_update_result(text, result, total):
    logger.info(text)
    for r in result:
        if r[1] == "add":
            logger.info(f"{r[1]} [{r[0]}]")
    for r in result:
        if r[1] == "delete":
            logger.info(f"{r[1]} [{r[0]}]")
    logger.info(f"Total : {total}")

def show_lora_key(json_path):
    data = {}

    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    for i, key in enumerate(data.keys()):
        logger.info(f"[{i}] {key}")



###############################################################

def update_lora_command(is_overwrite):

    logger.info(f"{is_overwrite=}")

    r = config_get_lora_dir_env_root_path()

    lora_path = config_get_character_lora_dir_path()
    json_path = r / Path("character_lora.json")
    result1,total1 = update_lora(lora_path, json_path, is_overwrite)

    lora_path = config_get_style_lora_dir_path()
    json_path = r / Path("style_lora.json")
    result2,total2 = update_lora(lora_path, json_path, is_overwrite)

    lora_path = config_get_pose_lora_dir_path()
    json_path = r / Path("pose_lora.json")
    result3,total3 = update_lora(lora_path, json_path, is_overwrite)

    lora_path = config_get_item_lora_dir_path()
    json_path = r / Path("item_lora.json")
    result4,total4 = update_lora(lora_path, json_path, is_overwrite)

    env = config_get_current_lora_dir_env()
    logger.info(f"[lora env = {env}]")
    print_update_result("== character lora ==", result1,total1)
    print_update_result("== style lora ==", result2,total2)
    print_update_result("== pose lora ==", result3,total3)
    print_update_result("== item lora ==", result4,total4)


def show_lora_command():
    r = config_get_lora_dir_env_root_path()
    env = config_get_current_lora_dir_env()
    logger.info(f"[lora env = {env}]")

    logger.info(f"== character lora ==")
    show_lora_key(r / Path("character_lora.json"))
    logger.info(f"== style lora ==")
    show_lora_key(r / Path("style_lora.json"))
    logger.info(f"== pose lora ==")
    show_lora_key(r / Path("pose_lora.json"))
    logger.info(f"== item lora ==")
    show_lora_key(r / Path("item_lora.json"))

def show_lora_env_command():
    envs = config_get_lora_dir_env_list()
    for i, ev in enumerate(envs):
        logger.info(f"{i} : {ev}")

    env = config_get_current_lora_dir_env()
    logger.info(f"current [lora env = {env}]")

def set_lora_env_command(new_env):
    env = config_get_current_lora_dir_env()
    logger.info(f"[lora env = {env}]")

    envs = config_get_lora_dir_env_list()
    if new_env not in envs:
        raise ValueError(f"{new_env=} is Not listed in lora_dir.json")

    config_set_current_lora_dir_env(new_env)

    env = config_get_current_lora_dir_env()
    logger.info(f"-> [lora env = {env}]")


def get_thumb_path(lora_type:LoraType, lora):

    if lora_type == LoraType.Character:
        lora_path = config_get_character_lora_dir_path()
    elif lora_type == LoraType.Style:
        lora_path = config_get_style_lora_dir_path()
    elif lora_type == LoraType.Pose:
        lora_path = config_get_pose_lora_dir_path()
    elif lora_type == LoraType.Item:
        lora_path = config_get_item_lora_dir_path()
    
    lora_path = lora_path / Path(lora)
    img_path = lora_path.with_suffix(".preview.png")
    return img_path

def get_lora_files_and_thumbs(lora_type:LoraType, thumb_size):

    lora_obj = Lora.create_instance(lora_type)

    file_list = lora_obj.get_file_list()
    lora_dir = lora_obj.get_file_dir()

    result = []
    for f in file_list:
        p = Path(lora_dir) / Path(f)

        if p.is_file():
            thumb = None
            thumb_path = p.with_suffix(".preview.png")
            if thumb_path.is_file():
                thumb = get_thumb(thumb_path, thumb_size)
            
            result.append((p.stem, thumb))

    return result

def get_lora_files_and_preview_paths(lora_type:LoraType):

    lora_obj = Lora.create_instance(lora_type)

    file_list = lora_obj.get_file_list()
    lora_dir = lora_obj.get_file_dir()

    result = []
    for f in file_list:
        p = Path(lora_dir) / Path(f)

        if p.is_file():
            thumb = None
            thumb_path = p.with_suffix(".preview.png")
            if thumb_path.is_file():
                thumb = thumb_path
            
            result.append((p.stem, thumb))

    return result

