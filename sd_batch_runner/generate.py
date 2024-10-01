import json
import logging
import time
from pathlib import Path
from datetime import datetime
import math
import random
from copy import deepcopy
from PIL import Image, PngImagePlugin
from enum import Enum
import shutil
import traceback

import webuiapi

from sd_batch_runner.util import *
from sd_batch_runner.lora import Lora,LoraType
from sd_batch_runner.sdwebui_temp_fix import ControlNetUnit2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



SD_HOST = "127.0.0.1"
SD_PORT = 7860




DEFAULT_GENERATION_COMMON_SETTING={
    "prompt":{
        "character_lora" : "@random"
    }
}

DEFAULT_GENERATION_SEQ_SETTING = [{
    "type":"txt2img",

}]


class RandomPicker():
    def __init__(self, common_rule, seq_rule):
        def to_list(a):
            return a if type(a) == list else [a,None]

        self.common_rule = to_list(common_rule)
        self.seq_rule = [to_list(a) for a in seq_rule]
        self._validate()

        if self.common_rule[0] in ("@random_once", "@random_per_seq", "@random"):
            #self.common_value = self._select_one()
            self.common_value = None
        else:
            self.common_value = self._select( self.common_rule[0], self.common_rule[1] )
        
        
        self.seq_value = []
        for sr in self.seq_rule:
            if sr[0] in ("@random_once", "@random_per_seq", "@random"):
                #self.seq_value.append(self._select_one())
                self.seq_value.append( None )
            else:
                self.seq_value.append(self._select( sr[0], sr[1] ))
    
    def _validate(self):
        for i in range(len(self.seq_rule)):
            if self.seq_rule[i][0] == "@random_per_seq":
                self.seq_rule[i][0] = "@random"

    def pick(self, index):

        def update(rule, index, v):
            if rule[0] == "@random_once":
                if v is not None:
                    return v
                else:
                    return self._select_one( rule[1] )
            elif rule[0] == "@random_per_seq":
                if index == 0:
                    return self._select_one( rule[1] )
            elif rule[0] == "@random":
                return self._select_one( rule[1] )
            
            return v
        
        self.common_value = update( self.common_rule, index, self.common_value)

        self.seq_value[index] = update( self.seq_rule[index], index, self.seq_value[index])

        seq_v = self.seq_value[index]

        if seq_v is None:
            return self.common_value
        
        return seq_v

class SeedPicker(RandomPicker):
    def __init__(self, common_rule, seq_rule):
        super().__init__(common_rule, seq_rule)
    
    def _select_one(self, opt):
        return int(random.randrange(4294967294))
    
    def _select(self, v, opt):
        if v == None:
            return None
        return int(v)

class LoraPicker(RandomPicker):
    def __init__(self, lora_type:LoraType, common_rule, seq_rule):
        self.lora_type = lora_type
        super().__init__(common_rule, seq_rule)

    def _adjust_weight(self, picked, opt):
        if picked is not None:
            if opt is not None:
                stem, weight, trigger = picked
                picked = [stem, weight * opt, trigger]
        return picked
    
    def _select_one(self, opt):
        filter = ""
        lora_str = opt
        if type(opt) == list:
            if len(opt) > 1:
                filter = opt[1]
            lora_str = opt[0]

        picked = Lora.select_one(self.lora_type, filter)
        return self._adjust_weight(picked, lora_str)
    
    def _select(self, v, opt):
        picked = Lora.select(self.lora_type, v)
        return self._adjust_weight(picked, opt)
    
    def get_type(self):
        return self.lora_type

class LoraAllPicker():
    def __init__(self, common_rule, seq_rule):
        def to_list(a):
            return a if type(a) == list else [a,None]
        def validate(a):
            return [] if a == None else a

        common_rule = validate(common_rule)
        seq_rule = [validate(s) for s in seq_rule]

        self.common_rule = [to_list(c) for c in common_rule] 
        self.seq_rule = [[to_list(a) for a in seq] for seq in seq_rule]

        self.common_value = [self._select( c[0], c[1] ) for c in self.common_rule]
        self.seq_value = [[self._select( s[0], s[1] ) for s in seq] for seq in self.seq_rule]

        logger.info(f"{self.common_value=}")
        logger.info(f"{self.seq_value=}")

    def _adjust_weight(self, picked, opt):
        if picked is not None:
            if opt is not None:
                stem, weight, trigger = picked
                picked = [stem, weight * opt, trigger]
        return picked
    
    def _select(self, v, opt):
        picked = Lora.select(LoraType.All, v)
        return self._adjust_weight(picked, opt)
        
    def pick(self, index):
        seq_v = self.seq_value[index]
        if seq_v in (None, []):
            return self.common_value
        
        return seq_v
        

class RandomPromptPicker(RandomPicker):
    def __init__(self, sd, common_rule, seq_rule):
        def convert(a):
            table={
                "once" : "@random_once",
                "per_seq" : "@random_per_seq",
                "any_time" : "@random"
            }
            return table.get(a, None)

        common_rule = convert(common_rule)
        seq_rule = [convert(s) for s in seq_rule]

        self.sd = sd
        super().__init__(common_rule, seq_rule)
    
    def pick(self, index, gen_setting):
        self.gen_setting = gen_setting
        result = super().pick(index)

        if result is None:
            return [None,None]

        lines = result.strip().split("Negative prompt")
        prompt = lines[0]
        if len(lines) > 1:
            negative_prompt = lines[1]
            negative_prompt = negative_prompt[16:].strip()
        else:
            negative_prompt = ""

        return [ prompt , negative_prompt]

    def _select_one(self, opt):
        return self.sd.prompt_gen(self.gen_setting)
    
    def _select(self, v, opt):
        return v



class PresetTags():
    def __init__(self, common_rule, seq_rule):
        self.common_rule = common_rule
        self.seq_rule = seq_rule
    def _apply(self, preset_name, prompt, neg_prompt):
        preset_info = config_get_preset_tags_info(preset_name)

        if preset_info["is_footer"]:
            prompt_list = [ prompt, preset_info["prompt"] ]
            neg_prompt_list = [ neg_prompt, preset_info["negative_prompt"] ]
        else:
            prompt_list = [ preset_info["prompt"], prompt ]
            neg_prompt_list = [ preset_info["negative_prompt"], neg_prompt ]

        if preset_info["prompt"]:
            prompt = ",".join(prompt_list)
        if preset_info["negative_prompt"]:
            neg_prompt = ",".join(neg_prompt_list)
        
        return prompt, neg_prompt

    def apply(self,index, prompt, neg_prompt):
        seq = self.seq_rule[index]
        cur = seq if seq!=None else self.common_rule

        if cur:
            for tag in cur:
                prompt, neg_prompt = self._apply(tag, prompt, neg_prompt)
        
        return prompt, neg_prompt


class InputSource():

    static_seq = 0
    static_instance_map={}

    class InnerInput():
        def __init__(self, dir_path:Path, rule):
            self.seq = -1
            self.dir_path = dir_path
            self.rule = rule
            self.cache = None       #[seq,index,img]
        
        def pick(self,index):
            if self.cache:
                if self.cache[0] == -1:
                    pass
                elif self.cache[0] != InputSource.static_seq:
                    self.cache = None
                elif self.rule == "@random":
                    if self.cache[1] != index:
                        self.cache = None

            if self.cache is None:
                img = self.get_image()
                if self.rule == "@random_once":
                    self.cache = [-1,-1,img]
                else:
                    self.cache = [InputSource.static_seq,index,img]
            
            return self.cache[2]
    
    class InnerInputDir(InnerInput):
        def __init__(self, dir_path:Path, rule, is_random):
            super().__init__(dir_path, rule)
            self.is_random = is_random
        def get_image(self):
            #return Image.open(select_one_file(self.dir_path, [".jpg",".png",".JPG",".PNG"], is_random=self.is_random))
            return select_one_file(self.dir_path, [".jpg",".png",".JPG",".PNG"], is_random=self.is_random)
        
    class InnerInputMov(InnerInput):
        def __init__(self, dir_path:Path, interval):
            super().__init__(dir_path, "@random")
            self.interval = interval
        def get_image(self):
            return select_frame(self.dir_path, self.interval)

    
    @classmethod
    def update_seq(cls, seq):
        InputSource.static_seq = seq
    
    @classmethod
    def pick(cls, index, dir_path:Path, is_random=True, rule="@random"):
        key = dir_path
        if key not in InputSource.static_instance_map:
            InputSource.static_instance_map[dir_path] = InputSource.InnerInputDir(dir_path, rule, is_random)
        
        return InputSource.static_instance_map[dir_path].pick(index)

    @classmethod
    def pick_m(cls, index, dir_path:Path, interval):
        key = dir_path
        if key not in InputSource.static_instance_map:
            InputSource.static_instance_map[dir_path] = InputSource.InnerInputMov(dir_path, interval)
        
        return InputSource.static_instance_map[dir_path].pick(index)


class PromptGenerator():
    def __init__(self, common_info, seq_info, sd):
        self.common_prompt =common_prompt = common_info.get("prompt", {})
        self.seq_prompt = seq_prompt = [ seq.get("prompt", {}) for seq in seq_info ]

        self.lora_order = common_prompt.get("lora_order", [])

        common_rule = common_prompt.get("character_lora", None)
        seq_rule = [ seq.get("character_lora", None) for seq in seq_prompt ]
        logger.info(f"CharacterLora {common_rule=} {seq_rule=}")
        self.character_lora = [LoraPicker( LoraType.Character, common_rule, seq_rule )]

        common_rule = common_prompt.get("character_lora2", None)
        seq_rule = [ seq.get("character_lora2", None) for seq in seq_prompt ]
        logger.info(f"CharacterLora 2 {common_rule=} {seq_rule=}")
        self.character_lora.append(LoraPicker( LoraType.Character, common_rule, seq_rule ))

        common_rule = common_prompt.get("style_lora", None)
        seq_rule = [ seq.get("style_lora", None) for seq in seq_prompt ]
        logger.info(f"StyleLora {common_rule=} {seq_rule=}")
        self.style_lora = [LoraPicker( LoraType.Style, common_rule, seq_rule )]

        common_rule = common_prompt.get("style_lora2", None)
        seq_rule = [ seq.get("style_lora2", None) for seq in seq_prompt ]
        logger.info(f"StyleLora 2 {common_rule=} {seq_rule=}")
        self.style_lora.append(LoraPicker( LoraType.Style, common_rule, seq_rule ))

        common_rule = common_prompt.get("pose_lora", None)
        seq_rule = [ seq.get("pose_lora", None) for seq in seq_prompt ]
        logger.info(f"PoseLora {common_rule=} {seq_rule=}")
        self.pose_lora = [LoraPicker( LoraType.Pose, common_rule, seq_rule )]

        common_rule = common_prompt.get("pose_lora2", None)
        seq_rule = [ seq.get("pose_lora2", None) for seq in seq_prompt ]
        logger.info(f"PoseLora 2 {common_rule=} {seq_rule=}")
        self.pose_lora.append(LoraPicker( LoraType.Pose, common_rule, seq_rule ))

        common_rule = common_prompt.get("item_lora", None)
        seq_rule = [ seq.get("item_lora", None) for seq in seq_prompt ]
        logger.info(f"ItemLora {common_rule=} {seq_rule=}")
        self.item_lora = [LoraPicker( LoraType.Item, common_rule, seq_rule )]

        common_rule = common_prompt.get("item_lora2", None)
        seq_rule = [ seq.get("item_lora2", None) for seq in seq_prompt ]
        logger.info(f"ItemLora 2 {common_rule=} {seq_rule=}")
        self.item_lora.append(LoraPicker( LoraType.Item, common_rule, seq_rule ))

        common_rule = common_prompt.get("additional_loras", None)
        seq_rule = [ seq.get("additional_loras", None) for seq in seq_prompt ]
        self.add_lora = LoraAllPicker( common_rule, seq_rule )

        common_rule = common_prompt.get("preset_tags", None)
        seq_rule = [ seq.get("preset_tags", None) for seq in seq_prompt ]
        self.preset_tags = PresetTags( common_rule, seq_rule )

        self.common_prompt_gen = common_prompt_gen = common_info.get("prompt_gen", {})
        self.seq_prompt_gen = seq_prompt_gen = [ seq.get("prompt_gen", {}) for seq in seq_info ]

        common_rule = common_prompt_gen.get("type", None)
        seq_rule = [ seq.get("type", None) for seq in seq_prompt_gen ]

        self.random_prompt_picker = RandomPromptPicker(sd, common_rule, seq_rule)

    def _append_random_prompt(self, index, prompt, neg_prompt):
        seq_setting = self.seq_prompt_gen[index]

        gen_setting = config_get_default_prompt_gen_setting()
        for s in self.common_prompt_gen:
            gen_setting[s] = self.common_prompt_gen[s]

        for s in seq_setting:
            gen_setting[s] = seq_setting[s]
        is_footer = gen_setting.get("is_footer", True)

        for k in ("type","is_footer"):
            if k in gen_setting:
                gen_setting.pop(k)
        
        pro,neg = self.random_prompt_picker.pick(index, gen_setting)

        if pro:
            if is_footer:
                prompt = ", ".join([prompt, pro])
                neg_prompt = ", ".join([neg_prompt, neg])
            else:
                prompt = ", ".join([pro, prompt])
                neg_prompt = ", ".join([neg, neg_prompt])
        
        return prompt, neg_prompt


    def generate(self, index, org_neg, base_part=True, lora_part=True, random_part=True):
        seq_prompt = self.seq_prompt[index]

        header = self.common_prompt.get("header", None)
        seq_header = seq_prompt.get("header", None)
        if seq_header:
            header = seq_header
        
        footer = self.common_prompt.get("footer", None)
        seq_footer = seq_prompt.get("footer", None)
        if seq_footer:
            footer = seq_footer
        
        prompt = ""
        if base_part:
            if header:
                prompt += header
        
        if lora_part:
            picked_list = []
            for i,lora in enumerate(( self.style_lora[0], self.style_lora[1], self.character_lora[0], self.character_lora[1], self.pose_lora[0], self.pose_lora[1], self.item_lora[0], self.item_lora[1] )):
                picked = lora.pick(index)
                if picked:
                    picked_list.append((i, picked, lora.get_type()))
            
            lora_w_rate = max( 1.0 * 0.9 ** (len(picked_list)-1) , 0.75)

            label = ["style_lora","style_lora2","character_lora","character_lora2","pose_lora","pose_lora2","item_lora","item_lora2"]

            if self.lora_order:
                lora_order = [label.index(o) for o in self.lora_order]
                custom_order = {c:i for i,c in enumerate(lora_order)}
                picked_list.sort(key=lambda c:custom_order.get(c[0], len(custom_order)))

            for i, picked, lora_type in picked_list:
                logger.info(f"{label[i]} = {picked[0]}")
                lora_index = 1 if label[i].endswith("2") else 0
                lora_syntax = create_lora_syntax(lora_type, lora_index, picked[0], picked[1] * lora_w_rate)
                if picked[2]:
                    prompt = ", ".join([prompt, lora_syntax, picked[2]])
                else:
                    prompt = ", ".join([prompt, lora_syntax])
            
            add_loras = self.add_lora.pick(index)
            for picked in add_loras:
                logger.info(f"add : {picked[0]}")
                lora_syntax = create_lora_syntax(LoraType.All, 0, picked[0], picked[1] * lora_w_rate)
                if picked[2]:
                    prompt = ", ".join([prompt, lora_syntax, picked[2]])
                else:
                    prompt = ", ".join([prompt, lora_syntax])


        if base_part:
            if footer:
                prompt = ", ".join([prompt, footer])
        
        neg_prompt = org_neg

        if random_part:
            prompt, neg_prompt = self._append_random_prompt(index, prompt, neg_prompt)

        if base_part:
            prompt, neg_prompt = self.preset_tags.apply(index, prompt, neg_prompt)

        return prompt, neg_prompt


class GenerationType(str, Enum):
    Txt2Img = "txt2img"
    Img2Img = "img2img"
    Copy = "copy"



class FocusUtility:
    def __init__(self, sd):
        self.sd = sd
        self.focus_cache = {}

    def get_focus_image_from_path(self, image_path, focus_target, scale):
        key = (image_path, focus_target, scale)
        if key not in self.focus_cache:

            org_img = Image.open(image_path)

            img = self.sd.create_focus_image(org_img, focus_target, scale)

            self.focus_cache[key] = img
        
        return self.focus_cache[key]
        
    def get_focus_image_from_image(self, org_img, key_index, focus_target, scale):
        key = (key_index, focus_target, scale)
        if key not in self.focus_cache:

            img = self.sd.create_focus_image(org_img, focus_target, scale)

            self.focus_cache[key] = img

        return self.focus_cache[key]

    def clear_cache(self):
        self.focus_cache = {}


def create_lora_syntax(lora_type:LoraType, lora_index, stem, weight):
    enable_lbw = False

    if lora_type == LoraType.Character:
        enable_lbw = config_get_lbw_enable_character(lora_index)
        preset = config_get_lbw_preset_character(lora_index)
        sss_type = config_get_lbw_start_stop_step_character(lora_index)
        sss_val = config_get_lbw_start_stop_step_value_character(lora_index)
    elif lora_type == LoraType.Style:
        enable_lbw = config_get_lbw_enable_style(lora_index)
        preset = config_get_lbw_preset_style(lora_index)
        sss_type = config_get_lbw_start_stop_step_style(lora_index)
        sss_val = config_get_lbw_start_stop_step_value_style(lora_index)
    elif lora_type == LoraType.Pose:
        enable_lbw = config_get_lbw_enable_pose(lora_index)
        preset = config_get_lbw_preset_pose(lora_index)
        sss_type = config_get_lbw_start_stop_step_pose(lora_index)
        sss_val = config_get_lbw_start_stop_step_value_pose(lora_index)
    elif lora_type == LoraType.Item:
        enable_lbw = config_get_lbw_enable_item(lora_index)
        preset = config_get_lbw_preset_item(lora_index)
        sss_type = config_get_lbw_start_stop_step_item(lora_index)
        sss_val = config_get_lbw_start_stop_step_value_item(lora_index)

    if enable_lbw:
        if sss_type in ("start","stop","step"):
            return f"<lora:{stem}:{weight:.2f}:{weight:.2f}:lbw={preset}:{sss_type}={sss_val}>"
        else:
            return f"<lora:{stem}:{weight:.2f}:{weight:.2f}:lbw={preset}>"
    else:
        return f"<lora:{stem}:{weight:.2f}>"


class GenerationSeqSetting():
    def __init__(self, generation_info, sd):
        self.gen_info = generation_info
        self.common = generation_info.get("common", DEFAULT_GENERATION_COMMON_SETTING)
        self.seq = generation_info.get("seq", DEFAULT_GENERATION_SEQ_SETTING)

        common_seed = self.common.get("seed", "@random_per_seq")
        seq_seed = [ seq.get("seed", None) for seq in self.seq ]
        self.seed = SeedPicker(common_seed, seq_seed)

        self.prompt = PromptGenerator(self.common, self.seq, sd)

        self.focus_util = FocusUtility(sd)

    def get_checkpoint_name(self):
        ck = config_get_default_checkpoint()
        gen_ck = self.common.get("checkpoint", "")
        if gen_ck:
            ck = gen_ck
        return ck
    
    def _extract_geninfo_from_image(self, index, image, ow_from_png, gen_setting):
        param = image.info.get('parameters',{})
        if not param:
            raise ValueError(f"invalid gen source")
        
        param = parse_generation_parameters(param)

        ow_flag = ow_from_png.get("overwrite_steps", True)
        if ow_flag:
            gen_setting["steps"] = int(param.get("Steps", gen_setting["steps"]))

        ow_flag = ow_from_png.get("overwrite_sampler_name", True)
        if ow_flag:
            gen_setting["sampler_name"] = param.get("Sampler", gen_setting["sampler_name"])

        ow_flag = ow_from_png.get("overwrite_scheduler", True)
        if ow_flag:
            gen_setting["scheduler"] = param.get("Schedule type", gen_setting["scheduler"])

        ow_flag = ow_from_png.get("overwrite_cfg_scale", True)
        if ow_flag:
            gen_setting["cfg_scale"] = float(param.get("CFG scale", gen_setting["cfg_scale"]))

        ow_flag = ow_from_png.get("overwrite_seed", True)
        if ow_flag:
            gen_setting["seed"] = int(param.get("Seed", gen_setting["seed"]))

        ow_flag = ow_from_png.get("overwrite_width", True)
        if ow_flag:
            gen_setting["width"] = int(param.get("Size-1", gen_setting["width"]))

        ow_flag = ow_from_png.get("overwrite_height", True)
        if ow_flag:
            gen_setting["height"] = int(param.get("Size-2", gen_setting["height"]))

        ow_flag = ow_from_png.get("overwrite_prompt", True)
        if ow_flag:
            gen_setting["prompt"] = param.get("Prompt", "")
            add_lora = ow_from_png.get("add_lora", False)
            add_prompt_gen = ow_from_png.get("add_prompt_gen", False)
            if add_lora or add_prompt_gen:
                gen_setting["prompt"] += "," + self.prompt.generate(index, "", base_part=False, lora_part=add_lora, random_part=add_prompt_gen)[0]


        ow_flag = ow_from_png.get("overwrite_negative_prompt", True)
        if ow_flag:
            gen_setting["negative_prompt"] = param.get("Negative prompt", gen_setting["negative_prompt"])


        return gen_setting
    

    def _overwrite_generation_setting(self, index, gen_setting):
        ow_from_png = deepcopy(config_get_default_overwrite_generation_setting())
        ow_ow_from_png = self.common.get("overwrite_generation_setting", {})
        if ow_ow_from_png:
            for o in ow_ow_from_png:
                ow_from_png[o] = ow_ow_from_png[o]
        ow_ow_from_png = self.seq[index].get("overwrite_generation_setting",{})
        if ow_ow_from_png:
            for o in ow_ow_from_png:
                ow_from_png[o] = ow_ow_from_png[o]

        if ow_from_png:
            image_rule = ow_from_png.get("png_info", None)
        
        if image_rule == None:
            return gen_setting

        img = self.get_image_from_image_rule(index, image_rule)
        gen_setting = self._extract_geninfo_from_image(index, img, ow_from_png, gen_setting)

        return gen_setting



    def _create_setting(self, index):
        gen_type = GenerationType( self.seq[index].get("type","txt2img") )
        is_txt2img = (gen_type == GenerationType.Txt2Img)

        gen_setting = deepcopy(config_get_default_generation_setting(is_txt2img))
        ow_gen_setting = self.common.get("generation_setting",{})
        if ow_gen_setting:
            for o in ow_gen_setting:
                gen_setting[o] = ow_gen_setting[o]
        ow_gen_setting = self.seq[index].get("generation_setting",{})
        if ow_gen_setting:
            for o in ow_gen_setting:
                gen_setting[o] = ow_gen_setting[o]
        
        gen_setting["seed"] = self.seed.pick(index)

        gen_setting = self._overwrite_generation_setting(index, gen_setting)

        if gen_setting.get("prompt", None):
            pass
        else:
            pro,neg = self.prompt.generate(index, gen_setting["negative_prompt"])
            gen_setting["prompt"] = pro
            gen_setting["negative_prompt"] = neg

        output_scale = self.seq[index].get("output_scale", None)
        if output_scale is not None:
            gen_setting["width"] = int(gen_setting["width"] * output_scale)
            gen_setting["height"] = int(gen_setting["height"] * output_scale)


        cn_setting = self.seq[index].get("controlnet", [])

        ad_setting = config_get_adetailer_setting()
        common_ad_setting = self.common.get("adetailer",[])
        if common_ad_setting:
            ad_setting = common_ad_setting
        seq_ad_setting = self.seq[index].get("adetailer",[])
        if seq_ad_setting:
            ad_setting = seq_ad_setting

        return (gen_type, gen_setting, cn_setting, ad_setting)
    
    def get_image_from_image_rule(self, index, image_rule):

        focus_target = focus_scale = None

        if type(image_rule) == list:
            if type(image_rule[0]) == list:
                focus_target = image_rule[1]
                focus_scale = image_rule[2]
                image_rule = image_rule[0]
                if len(image_rule) == 1:
                    image_rule = image_rule[0]

        path_or_img = self.get_path_or_image_from_image_rule(index,image_rule)

        if isinstance(path_or_img, Image.Image):
            return path_or_img
        else:

            if focus_target:
                if isinstance(path_or_img, Path):
                    return self.focus_util.get_focus_image_from_path(path_or_img, focus_target, focus_scale)
                elif type(path_or_img) == int:
                    org_img = self.result[ path_or_img ]
                    return self.focus_util.get_focus_image_from_image(org_img, path_or_img, focus_target, focus_scale)
                else:
                    raise ValueError(f"unknown format {path_or_img=}")
            else:

                if isinstance(path_or_img, Path):
                    return Image.open(path_or_img)
                elif type(path_or_img) == int:
                    return self.result[ path_or_img ]
                else:
                    raise ValueError(f"unknown format {path_or_img=}")


    def get_path_or_image_from_image_rule(self, index, image_rule):
        if type(image_rule) == int:
            return image_rule       #self.result[ image_rule ]
        elif type(image_rule) == str:
            image_path = Path(image_rule)
            if image_path.is_file():
                return Path(image_rule)
            elif image_path.is_dir():
                return InputSource.pick(index, image_path)
            else:
                raise ValueError(f"unknown rule {image_rule=}")
        elif type(image_rule) == list:
            image_path = Path(image_rule[0])
            if image_path.is_dir():
                rule = "@random"
                if len(image_rule) > 2:
                    rule = image_rule[2]
                is_random = True
                if len(image_rule) > 1:
                    is_random = False if image_rule[1]=="sort" else True
                return InputSource.pick(index, image_path, is_random, rule)
            elif image_path.is_file() and image_path.suffix in (".mp4", ".MP4"):
                if type(image_rule[1]) in (int, float):
                    return InputSource.pick_m(index, image_path, image_rule[1])
                else:
                    raise ValueError(f"unknown rule {image_rule[1]=} must be int or float")
            else:
                raise ValueError(f"unknown rule {image_rule=} {image_path} is invalid path")
        else:
            raise ValueError(f"unknown rule {image_rule=}")

    def get_input_image(self, index):
        input_image_rule = self.seq[index].get("input_image", 0)
        return self.get_image_from_image_rule(index, input_image_rule)

    def get_input_image_or_path(self, index):
        input_image_rule = self.seq[index].get("input_image", 0)
        # ignore focus rule
        if type(input_image_rule) == list:
            if type(input_image_rule[0]) == list:
                input_image_rule = input_image_rule[0]
                if len(input_image_rule) == 1:
                    input_image_rule = input_image_rule[0]
                    
        return self.get_path_or_image_from_image_rule(index, input_image_rule)

    def set_result(self, index, image):
        self.result[index] = image

    def set_cn_mask_target(self, index, mask):
        self.cn_mask_target[index] = mask
    
    def get_cn_mask_target(self, index):
        return self.cn_mask_target[index]

    def __getitem__(self, index):
        if index == 0:
            self.result = [None for n in range(len(self.seq))]
            self.cn_mask_target = [None for n in range(len(self.seq))]
            self.focus_util.clear_cache()
        
        logger.info(f"{index=}")
        if 0 <= index < len(self.seq):
            try:
                return self._create_setting(index)
            except Exception as e:
                logger.error(traceback.format_exc())
                #exit()
                raise e
        else:
            raise IndexError(f"invalid {index=}")



class SDGen:
    def __init__( self, host=SD_HOST, port=SD_PORT, output_dir_path=Path("output") ):
        self.api = webuiapi.WebUIApi(host=host, port=port)
        self.segif = webuiapi.SegmentAnythingInterface(self.api)
        self.output_dir_path = output_dir_path
    
    def get_checkpoints(self):
        return self.api.util_get_model_names()

    def get_latent_upscale_modes(self):
        return sorted([s['name'] for s in self.api.get_latent_upscale_modes()])

    def get_samplers(self):
        return self.api.util_get_sampler_names()
    
    def get_schedulers(self):
        return self.api.util_get_scheduler_names()
    
    def get_sam_models(self):
        return self.segif.get_sam_models()

    def get_loras(self):
        return self.api.get_loras()
    
    def get_controlnets(self):
        return self.api.controlnet_model_list(), self.api.controlnet_module_list()
    
    def set_checkpoint(self, name):
        self.api.util_set_model(name)
    
    def prompt_gen(self, prompt_gen_setting):
        prompt_gen_setting["batch_count"] = 1
        prompt_gen_setting["batch_size"] = 1
        result = self.api.prompt_gen(**prompt_gen_setting)
        return result[0]
    
    def setup_seq( self, generation_info ):
        self.gen_seq = GenerationSeqSetting(generation_info, self)

    def get_seq_length(self):
        return len(self.gen_seq.seq)
    
    def get_mask_of_image(self, img, mask_target):

        if type(mask_target) == int:
            return self.gen_seq.get_cn_mask_target(mask_target)
        
        try:
            sam_result = self.segif.sam_predict(
                image=img,
                sam_model_name = config_get_segment_anything_sam_model_name(),
                dino_enabled=True,
                dino_text_prompt=mask_target,
                dino_model_name= config_get_segment_anything_dino_model_name()
                )
            
            dilation_result = self.segif.dilate_mask(
                image=img,
                mask=sam_result.masks[0],  # using the first mask from the SAM prediction
                dilate_amount=30
                )
        except Exception as e:
            return None
        
        return dilation_result.mask.convert('RGB')

    def create_focus_image(self, img, focus_target, scale):

        try:
            sam_result = self.segif.sam_predict(
                image=img,
                sam_model_name = config_get_segment_anything_sam_model_name(),
                dino_enabled=True,
                dino_text_prompt=focus_target,
                dino_model_name= config_get_segment_anything_dino_model_name()
                )
            
            dilation_result = self.segif.dilate_mask(
                image=img,
                mask=sam_result.masks[0],  # using the first mask from the SAM prediction
                dilate_amount=30
                )
            
            focus_point = get_center_of_mask(dilation_result.mask, 0)
        except Exception as e:
            focus_point = None

        focus_img = create_focus_image(img, focus_point, scale)

        return focus_img
    
    def create_controlnet_units(self, index, cn_setting):
        units = []
        for cn in cn_setting:
            name = cn["type"]
            s = get_controlnet_setting(name)
            for key in cn.keys():
                if key not in ("type","image","mask","image_scale","cn_target"):
                    s[key] = cn[key]

            #unit = webuiapi.ControlNetUnit(**s)
            unit = ControlNetUnit2(**s)
            img = cn.get("image", None)
            if img is not None:
                unit.image = self.gen_seq.get_image_from_image_rule(index, img)
                img_scale = cn.get("image_scale", None)
                if img_scale:
                    new_size = (int(unit.image.size[0] * img_scale)//8 * 8, int(unit.image.size[1] * img_scale)//8 * 8)
                    unit.image = unit.image.resize( new_size )
            
            def is_valid_target(cn_t):
                if type(cn_t) in (int,float):
                    return True
                else:
                    return cn_t

            cn_target = cn.get("cn_target", None)
            if is_valid_target(cn_target) and (img is not None):
                unit.effective_region_mask = self.get_mask_of_image(unit.image, cn_target)
                self.gen_seq.set_cn_mask_target(index, unit.effective_region_mask)

            img = cn.get("mask", None)
            if img is not None:
                unit.mask = self.gen_seq.get_image_from_image_rule(index, img)

            units.append(unit)
        
        return units
    
    def create_adetailer_units(self, index, ad_setting):
        units = []
        for ad in ad_setting:
            unit = webuiapi.ADetailer(**ad)
            units.append(unit)
        return units

    def _run( self, index, gen_type:GenerationType, gen_setting, cn_setting, ad_setting):

        logger.info(f"{index=}")
        logger.info(f"{gen_type=}")
        logger.info(f"{gen_setting=}")
        logger.info(f"{cn_setting=}")
        logger.info(f"{ad_setting=}")

        if gen_type == GenerationType.Copy:
            copy_src = self.gen_seq.get_input_image_or_path(index)
            if isinstance(copy_src, Path):
                img = Image.open(copy_src)
                self.gen_seq.set_result(index, img)
                result = copy_src
            else:
                raise ValueError(f"copy source(input_image) invalid {copy_src=}")
        else:

            cn_units = self.create_controlnet_units(index, cn_setting)
            ad_units = self.create_adetailer_units(index, ad_setting)

            alwayson_scripts = {
                "Simple wildcards": []
            }

            if gen_type == GenerationType.Txt2Img:
                result = self.api.txt2img( **gen_setting, alwayson_scripts=alwayson_scripts, controlnet_units=cn_units, adetailer=ad_units)
            else:
                images = [ self.gen_seq.get_input_image(index) ]
                result = self.api.img2img( images=images, mask_image=None, **gen_setting, alwayson_scripts=alwayson_scripts, controlnet_units=cn_units, adetailer=ad_units)

            self.gen_seq.set_result(index, result.images[0])

        return result
    
    def generate( self, n = 1):

        ck = self.gen_seq.get_checkpoint_name()
        if ck:
            self.set_checkpoint( ck )

        for i in range(n):

            InputSource.update_seq(i)

            for f,s in enumerate(self.gen_seq):
                result = self._run(f, s[0], s[1], s[2], s[3])
                if isinstance(result, Path):
                    output_path = self.output_dir_path / Path( f"{str(i).zfill(5)}_{str(f).zfill(5)}.png")
                    shutil.copy(result, output_path)
                else:
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("parameters", result.info['infotexts'][0])
                    result.image.save( self.output_dir_path / Path( f"{str(i).zfill(5)}_{str(f).zfill(5)}.png" ) , pnginfo=pnginfo)

    def generate_generator( self, n = 1):

        ck = self.gen_seq.get_checkpoint_name()
        if ck:
            self.set_checkpoint( ck )

        for i in range(n):

            InputSource.update_seq(i)

            for f,s in enumerate(self.gen_seq):
                result = self._run(f, s[0], s[1], s[2], s[3])
                if isinstance(result, Path):
                    output_path = self.output_dir_path / Path( f"{str(i).zfill(5)}_{str(f).zfill(5)}.png")
                    shutil.copy(result, output_path)
                else:
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("parameters", result.info['infotexts'][0])
                    result.image.save( self.output_dir_path / Path( f"{str(i).zfill(5)}_{str(f).zfill(5)}.png" ) , pnginfo=pnginfo)
                
                yield



###############################################################

def one_command(char,style,pose,item,header,footer,n):

    time_str = get_time_str()
    output_dir = Path("output") / Path(time_str)
    output_dir.mkdir(parents=True)

    sd = SDGen(output_dir_path=output_dir)

    info = {
        "common" : {
            "prompt":{

            }
        }
    }

    if char:
        info["common"]["prompt"]["character_lora"] = char
    if style:
        info["common"]["prompt"]["style_lora"] = style
    if pose:
        info["common"]["prompt"]["pose_lora"] = pose
    if item:
        info["common"]["prompt"]["item_lora"] = item
    if header:
        info["common"]["prompt"]["header"] = header
    if footer:
        info["common"]["prompt"]["footer"] = footer

    sd.setup_seq(info)

    backup_json_path = Path(output_dir)/Path(time_str + "_gen.json")
    json_text = json.dumps(info, indent=4, ensure_ascii=False)
    backup_json_path.write_text(json_text, encoding="utf-8")

    sd.generate(n)


def generate_command(json_path:Path, n):

    time_str = get_time_str()
    output_dir = Path("output") / Path(time_str)
    output_dir.mkdir(parents=True)

    sd = SDGen(output_dir_path=output_dir)

    info = {}
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        raise ValueError(f"invalid path {json_path=}")

    sd.setup_seq(info)

    backup_json_path = Path(output_dir)/Path(time_str + "_gen.json")
    json_text = json.dumps(info, indent=4, ensure_ascii=False)
    backup_json_path.write_text(json_text, encoding="utf-8")

    sd.generate(n)


def show_checkpoint_command():
    sd = SDGen()
    for i,c in enumerate(sd.get_checkpoints()):
        logger.info(f"[{i}] {c}")

def set_default_checkpoint_command(checkpoint_number):
    sd = SDGen()
    cks = sd.get_checkpoints()

    if checkpoint_number >= len(cks):
        raise ValueError(f"invalid {checkpoint_number=}")
    
    config_set_default_checkpoint(cks[checkpoint_number])

    logger.info(f"Set to {config_get_default_checkpoint()}")

def show_controlnet_command():
    sd = SDGen()

    models, modules = sd.get_controlnets()
    logger.info(f"== Controlnet Models ==")
    for i,c in enumerate(models):
        logger.info(f"[{i}] {c}")
    logger.info(f"== Controlnet Modules ==")
    for i,c in enumerate(modules):
        logger.info(f"[{i}] {c}")


###############################################################
_controlnet_model_cache = []
_controlnet_module_cache = []

def get_controlnet_list():
    global _controlnet_model_cache,_controlnet_module_cache
    if not _controlnet_model_cache:
        sd = SDGen()
        _controlnet_model_cache,_controlnet_module_cache = sd.get_controlnets()
        _controlnet_model_cache = ["none"] + _controlnet_model_cache
    return _controlnet_model_cache,_controlnet_module_cache


###############################################################

_checkpoint_list_cache=[]
def get_checkpoint_list():
    global _checkpoint_list_cache
    if not _checkpoint_list_cache:
        sd = SDGen()
        _checkpoint_list_cache = sd.get_checkpoints()
    return _checkpoint_list_cache


_latent_upscale_mode_list_cache=[]
def get_latent_upscale_mode_list():
    global _latent_upscale_mode_list_cache
    if not _latent_upscale_mode_list_cache:
        sd = SDGen()
        _latent_upscale_mode_list_cache = sd.get_latent_upscale_modes()
    return _latent_upscale_mode_list_cache


_sampler_list_cache=[]
def get_sampler_list():
    global _sampler_list_cache
    if not _sampler_list_cache:
        sd = SDGen()
        _sampler_list_cache = sd.get_samplers()
    return _sampler_list_cache


_scheduler_list_cache=[]
def get_scheduler_list():
    global _scheduler_list_cache
    if not _scheduler_list_cache:
        sd = SDGen()
        _scheduler_list_cache = sd.get_schedulers()
    return _scheduler_list_cache


_sam_model_list_cache=[]
def get_sam_model_list():
    global _sam_model_list_cache
    if not _sam_model_list_cache:
        sd = SDGen()
        _sam_model_list_cache = sd.get_sam_models()
    return _sam_model_list_cache




###############################################################
generate_cancel_flag = False

def sd_task(output_dir, info, n, on_progress, on_complete):

    sd = SDGen(output_dir_path=output_dir)

    sd.setup_seq(info)

    progress = 0
    total = sd.get_seq_length() * n

    g = sd.generate_generator(n)

    for progress in range(1, total+1):
        try:
            next(g)
        except Exception as e:
            logger.error(traceback.format_exc())
            on_complete("Failed")
            return
        
        on_progress(progress/total)
        if generate_cancel_flag:
            on_complete("Cancel")
            return

    on_complete("Success")


def async_generate(json_path:Path, n, on_progress, on_complete):
    import threading

    global generate_cancel_flag

    time_str = get_time_str()
    output_dir = Path("output") / Path(time_str)
    output_dir.mkdir(parents=True)

    info = {}
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        raise ValueError(f"invalid path {json_path=}")

    backup_json_path = Path(output_dir)/Path(time_str + "_gen.json")
    json_text = json.dumps(info, indent=4, ensure_ascii=False)
    backup_json_path.write_text(json_text, encoding="utf-8")

    generate_cancel_flag = False

    thr = threading.Thread(target=sd_task, args=(output_dir, info, n, on_progress, on_complete))
    thr.start()

    return output_dir.absolute()

def cancel_generate():
    global generate_cancel_flag
    generate_cancel_flag = True

