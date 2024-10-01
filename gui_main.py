import json
import logging
from pathlib import Path
from enum import Enum
from copy import deepcopy

import flet as ft

from sd_batch_runner.generate import get_controlnet_list,get_checkpoint_list,get_latent_upscale_mode_list,get_sampler_list,get_scheduler_list,get_sam_model_list,async_generate,cancel_generate
from sd_batch_runner.lora import update_lora_command,LoraType, lora_clear_cache
from sd_batch_runner.util import get_time_str


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



def validate_image_rule(image_rule):
    if type(image_rule) == int:
        return ("generation_result", image_rule)
    elif type(image_rule) == str:
        image_path = Path(image_rule)
        if image_path.is_file():
            return ("image_file", image_rule)
        elif image_path.is_dir():
            return ("image_directory", [image_rule, "random" , "@random"])
        else:
            return ("image_file", image_rule)
            #raise ValueError(f"unknown rule {image_rule=}")
    elif type(image_rule) == list:
        image_path = Path(image_rule[0])
        #if image_path.is_file() and image_path.suffix in (".mp4", ".MP4"):
        if image_path.suffix in (".mp4", ".MP4"):
            if type(image_rule[1]) in (int, float):
                return ("movie_file", image_rule)
            else:
                raise ValueError(f"unknown rule {image_rule[1]=} must be int or float")
        #elif image_path.is_dir():
        else:
            sort_rule = "random"
            rule = "@random"

            if len(image_rule) == 3:
                sort_rule = image_rule[1]
                rule = image_rule[2]
            elif len(image_rule) == 2:
                rule = image_rule[1]

            return ("image_directory", [image_rule[0], sort_rule , rule])
        #else:
        #    raise ValueError(f"unknown rule {image_rule=}")
        
    else:
        raise ValueError(f"unknown rule {image_rule=}")


def validate_focus_rule(image_rule):
    focus_target = ""
    focus_scale = 1.0
    
    if type(image_rule) == list:
        if type(image_rule[0]) == list:
            focus_target = image_rule[1]
            focus_scale = image_rule[2]
            image_rule = image_rule[0]
            if len(image_rule) == 1:
                image_rule = image_rule[0]
    
    return image_rule, focus_target, focus_scale


def get_controlnet_type_list():
    json_path = Path("controlnet.json")

    info = {}
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        logger.info(f"{json_path} not found")

    return list(info.keys())


def get_lora_dir_env_list():
    json_path = Path("lora_dir.json")

    info = {}
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        logger.info(f"{json_path} not found")

    return list(info.keys())

def get_preset_tags_list():
    json_path = Path("preset_tags.json")

    info = {}
    if json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        logger.info(f"{json_path} not found")

    return list(info.keys())


def str_to_bool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def str_to_token(val):
    return val.split(',')

class ConfigType(str, Enum):
    Controlnet_Module = "controlnet_module",
    Controlnet_Model = "controlnet_model",
    Controlnet_Weight = "controlnet_weight",
    Controlnet_ResizeMode = "controlnet_resize_mode",
    Controlnet_LowVram = "controlnet_low_vram",
    Controlnet_ProcessorRes = "controlnet_processor_res",
    Controlnet_Tha = "controlnet_threshold_a",
    Controlnet_Thb = "controlnet_threshold_b",
    Controlnet_GuidanceStart = "controlnet_guidance_start",
    Controlnet_GuidanceEnd = "controlnet_guidance_end",
    Controlnet_ControlMode = "controlnet_control_mode",
    Controlnet_PixelPerfect = "controlnet_pixel_perfect",
    Controlnet_HrOption = "controlnet_hr_option",

    LoraEnv_CharacterDir = "loraenv_character_dir_path",
    LoraEnv_StyleDir = "loraenv_style_dir_path",
    LoraEnv_PoseDir = "loraenv_pose_dir_path",
    LoraEnv_ItemDir = "loraenv_item_dir_path",

    PresetTags_Prompt = "preset_tags_prompt",
    PresetTags_NegPrompt = "preset_tags_negative_prompt",
    PresetTags_IsFooter = "preset_tags_is_footer",

    DefaultCheckPoint = "default_checkpoint",
    LoraEnv = "lora_dir_env"

    LoraGenerateTag_EnableCharacter = "lora_generate_tag_enable_character",
    LoraGenerateTag_EnableStyle = "lora_generate_tag_enable_style",
    LoraGenerateTag_EnablePose = "lora_generate_tag_enable_pose",
    LoraGenerateTag_EnableItem = "lora_generate_tag_enable_item",
    LoraGenerateTag_TagThCharacter = "lora_generate_tag_tag_th_character",
    LoraGenerateTag_TagThStyle = "lora_generate_tag_tag_th_style",
    LoraGenerateTag_TagThPose = "lora_generate_tag_tag_th_pose",
    LoraGenerateTag_TagThItem = "lora_generate_tag_tag_th_item",
    LoraGenerateTag_ProhibitedTagsCharacter = "lora_generate_tag_prohibited_tags_character",
    LoraGenerateTag_ProhibitedTagsStyle = "lora_generate_tag_prohibited_tags_style",
    LoraGenerateTag_ProhibitedTagsPose = "lora_generate_tag_prohibited_tags_pose",
    LoraGenerateTag_ProhibitedTagsItem = "lora_generate_tag_prohibited_tags_item",

    LoraBlockWeight_EnableLbwCharacter = "lbw_enable_lbw_character",
    LoraBlockWeight_PresetCharacter = "lbw_preset_character",
    LoraBlockWeight_SssCharacter = "lbw_start_stop_step_character",
    LoraBlockWeight_SssValueCharacter = "lbw_start_stop_step_value_character",
    LoraBlockWeight_EnableLbwCharacter2 = "lbw_enable_lbw_character2",
    LoraBlockWeight_PresetCharacter2 = "lbw_preset_character2",
    LoraBlockWeight_SssCharacter2 = "lbw_start_stop_step_character2",
    LoraBlockWeight_SssValueCharacter2 = "lbw_start_stop_step_value_character2",

    LoraBlockWeight_EnableLbwStyle = "lbw_enable_lbw_style",
    LoraBlockWeight_PresetStyle = "lbw_preset_style",
    LoraBlockWeight_SssStyle = "lbw_start_stop_step_style",
    LoraBlockWeight_SssValueStyle = "lbw_start_stop_step_value_style",
    LoraBlockWeight_EnableLbwStyle2 = "lbw_enable_lbw_style2",
    LoraBlockWeight_PresetStyle2 = "lbw_preset_style2",
    LoraBlockWeight_SssStyle2 = "lbw_start_stop_step_style2",
    LoraBlockWeight_SssValueStyle2 = "lbw_start_stop_step_value_style2",

    LoraBlockWeight_EnableLbwPose = "lbw_enable_lbw_pose",
    LoraBlockWeight_PresetPose = "lbw_preset_pose",
    LoraBlockWeight_SssPose = "lbw_start_stop_step_pose",
    LoraBlockWeight_SssValuePose = "lbw_start_stop_step_value_pose",
    LoraBlockWeight_EnableLbwPose2 = "lbw_enable_lbw_pose2",
    LoraBlockWeight_PresetPose2 = "lbw_preset_pose2",
    LoraBlockWeight_SssPose2 = "lbw_start_stop_step_pose2",
    LoraBlockWeight_SssValuePose2 = "lbw_start_stop_step_value_pose2",

    LoraBlockWeight_EnableLbwItem = "lbw_enable_lbw_item",
    LoraBlockWeight_PresetItem = "lbw_preset_item",
    LoraBlockWeight_SssItem = "lbw_start_stop_step_item",
    LoraBlockWeight_SssValueItem = "lbw_start_stop_step_value_item",
    LoraBlockWeight_EnableLbwItem2 = "lbw_enable_lbw_item2",
    LoraBlockWeight_PresetItem2 = "lbw_preset_item2",
    LoraBlockWeight_SssItem2 = "lbw_start_stop_step_item2",
    LoraBlockWeight_SssValueItem2 = "lbw_start_stop_step_value_item2",

    GenTxt2Img_EnableHr = "txt2img_enable_hr",
    GenTxt2Img_DenoisingStr = "txt2img_denoising_strength",
    GenTxt2Img_1pWidth = "txt2img_firstphase_width",
    GenTxt2Img_1pHeight = "txt2img_firstphase_height",
    GenTxt2Img_HrScale = "txt2img_hr_scale",
    GenTxt2Img_HrUpscaler = "txt2img_hr_upscaler",
    GenTxt2Img_Hr2pSteps = "txt2img_hr_second_pass_steps",
    GenTxt2Img_HrResizeX = "txt2img_hr_resize_x",
    GenTxt2Img_HrResizeY = "txt2img_hr_resize_y",

    GenImg2Img_ResizeMode = "img2img_resize_mode",
    GenImg2Img_DenoisingStr = "img2img_denoising_strength",
    GenImg2Img_ImgCfgScale = "img2img_image_cfg_scale",
    GenImg2Img_MaskBlur = "img2img_mask_blur",
    GenImg2Img_InpaintingFill = "img2img_inpainting_fill",
    GenImg2Img_InpaintFullRes = "img2img_inpaint_full_res",
    GenImg2Img_InpaintFullResPadding = "img2img_inpaint_full_res_padding",
    GenImg2Img_InpaintingMaskInvert = "img2img_inpainting_mask_invert",
    GenImg2Img_InitialNoiseMul = "img2img_initial_noise_multiplier",

    CommonGen_SamplerName = "common_sampler_name",
    CommonGen_Scheduler = "common_scheduler",
    CommonGen_Steps = "common_steps",
    CommonGen_CfgScale = "common_cfg_scale",
    CommonGen_Width = "common_width",
    CommonGen_Height = "common_height",
    CommonGen_RestoreFaces = "common_restore_faces",
    CommonGen_Tiling = "common_tiling",
    CommonGen_DnSaveSamples = "common_do_not_save_samples",
    CommonGen_DnSaveGrid = "common_do_not_save_grid",
    CommonGen_NegPrompt = "common_negative_prompt",
    CommonGen_Eta = "common_eta",
    CommonGen_SendImages = "common_send_images",
    CommonGen_SaveImages = "common_save_images",

    PromptGen_ModelName = "promptgen_model_name",
    PromptGen_Text = "promptgen_text",
    PromptGen_MinLength = "promptgen_min_length",
    PromptGen_MaxLength = "promptgen_max_length",
    PromptGen_NumBeams = "promptgen_num_beams",
    PromptGen_Temp = "promptgen_temperature",
    PromptGen_RepPenalty = "promptgen_repetition_penalty",
    PromptGen_LengthPref = "promptgen_length_preference",
    PromptGen_SamplingMode = "promptgen_sampling_mode",
    PromptGen_TopK = "promptgen_top_k",
    PromptGen_TopP = "promptgen_top_p",

    OWGen_OWSteps = "owgen_overwrite_steps",
    OWGen_OWSamplerName = "owgen_overwrite_sampler_name",
    OWGen_OWScheduler = "owgen_overwrite_scheduler",
    OWGen_OWCfgScale = "owgen_overwrite_cfg_scale",
    OWGen_OWWidth = "owgen_overwrite_width",
    OWGen_OWHeight = "owgen_overwrite_height",
    OWGen_OWPrompt = "owgen_overwrite_prompt",
    OWGen_OWNegPrompt = "owgen_overwrite_negative_prompt",
    OWGen_OWSeed = "owgen_overwrite_seed",
    OWGen_AddLora = "owgen_add_lora",
    OWGen_AddPromptGen = "owgen_add_prompt_gen",

    Adetailer_Model = "adetailer_model",
    Adetailer_Prompt = "adetailer_prompt",
    Adetailer_NegPrompt = "adetailer_negative_prompt",
    Adetailer_Model2 = "adetailer_model2",
    Adetailer_Prompt2 = "adetailer_prompt2",
    Adetailer_NegPrompt2 = "adetailer_negative_prompt2",
    Adetailer_Model3 = "adetailer_model3",
    Adetailer_Prompt3 = "adetailer_prompt3",
    Adetailer_NegPrompt3 = "adetailer_negative_prompt3",

    Sam_SamModelName = "segment_anything_sam_model_name",
    Sam_DinoModelName = "segment_anything_dino_model_name",

    Input_Prompt_CharacterLora = "input_prompt_character_lora",
    Input_Prompt_CharacterLora2 = "input_prompt_character_lora2",
    Input_Prompt_StyleLora = "input_prompt_style_lora",
    Input_Prompt_StyleLora2 = "input_prompt_style_lora2",
    Input_Prompt_PoseLora = "input_prompt_pose_lora",
    Input_Prompt_PoseLora2 = "input_prompt_pose_lora2",
    Input_Prompt_ItemLora = "input_prompt_item_lora",
    Input_Prompt_ItemLora2 = "input_prompt_item_lora2",
    Input_Prompt_PresetTags = "input_prompt_preset_tags",
    Input_Prompt_Header = "input_prompt_header",
    Input_Prompt_Footer = "input_prompt_footer",

    Input_CheckPoint = "input_checkpoint"
    Input_Seed = "input_seed",

    Input_OW_PngInfo = "input_overwrite_pnginfo",

    Input_PromptGen_Type = "input_promptgen_type",
    Input_PromptGen_IsFooter = "input_promptgen_is_footer",

    Input_Seq_Type = "input_seq_type",
    Input_Seq_InputImage = "input_seq_input_image",
    Input_Seq_OutputScale = "input_seq_output_scale",

    Input_Seq_ControlnetType = "input_seq_controlnet_type",
    Input_Seq_ControlnetImage = "input_seq_controlnet_image",
    Input_Seq_ControlnetCNTarget = "input_seq_controlnet_cn_target",









DD_OPTION_MAP = {
    ConfigType.Controlnet_HrOption : ["Both","Low res only","High res only"],
    ConfigType.Controlnet_ResizeMode : ["Just Resize","Crop and Resize","Resize and Fill"],
    #"module" : controlnet_module_list,
    #"model" : controlnet_model_list,
    ConfigType.Controlnet_ControlMode : ["0","1","2"],

    ConfigType.LoraBlockWeight_SssCharacter : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssCharacter2 : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssStyle : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssStyle2 : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssPose : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssPose2 : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssItem : ["none","start","stop","step"],
    ConfigType.LoraBlockWeight_SssItem2 : ["none","start","stop","step"],

    #"hr_upscaler" : latent_upscale_mode_list,

    ConfigType.GenImg2Img_ResizeMode : ["0","1","2","3"],

    #"sampler_name": sampler_list,
    #"scheduler" : scheduler_list,

    ConfigType.PromptGen_ModelName : ["AUTOMATIC/promptgen-lexart", "AUTOMATIC/promptgen-majinai-safe", "AUTOMATIC/promptgen-majinai-unsafe", "promptgen-lexart", "promptgen-majinai-safe", "promptgen-majinai-unsafe"],
    ConfigType.PromptGen_SamplingMode : ["Top K","Top P"],

    ConfigType.Adetailer_Model : ["none", "face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt", "person_yolov8n-seg.pt", "person_yolov8s-seg.pt", "mediapipe_face_full", "mediapipe_face_short", "mediapipe_face_mesh" ],
    ConfigType.Adetailer_Model2 : ["none", "face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt", "person_yolov8n-seg.pt", "person_yolov8s-seg.pt", "mediapipe_face_full", "mediapipe_face_short", "mediapipe_face_mesh" ],
    ConfigType.Adetailer_Model3 : ["none", "face_yolov8n.pt", "face_yolov8s.pt", "hand_yolov8n.pt", "person_yolov8n-seg.pt", "person_yolov8s-seg.pt", "mediapipe_face_full", "mediapipe_face_short", "mediapipe_face_mesh" ],

    ConfigType.Input_Seed : ["@random_once","@random_per_seq","@random"],
    ConfigType.Input_PromptGen_Type : ["once","per_seq","any_time"],
    ConfigType.Input_Seq_Type : ["txt2img","img2img","copy"],

}

CONFIG_MAP = {
    # ui_type, format_func, data_name
    # 
    ConfigType.Controlnet_Module : ["dd", str, "module"],
    ConfigType.Controlnet_Model : ["dd", str, "model"],
    ConfigType.Controlnet_Weight : ["num", float, "weight"],
    ConfigType.Controlnet_ResizeMode : ["dd", str, "resize_mode" ],
    ConfigType.Controlnet_LowVram : ["bool", bool, "low_vram" ],
    ConfigType.Controlnet_ProcessorRes : ["num", float, "processor_res" ],
    ConfigType.Controlnet_Tha : ["num", float, "threshold_a" ],
    ConfigType.Controlnet_Thb : ["num", float, "threshold_b" ],
    ConfigType.Controlnet_GuidanceStart : ["num", float, "guidance_start" ],
    ConfigType.Controlnet_GuidanceEnd : ["num", float, "guidance_end" ],
    ConfigType.Controlnet_ControlMode : ["dd", int, "control_mode" ],
    ConfigType.Controlnet_PixelPerfect : ["bool", bool, "pixel_perfect" ],
    ConfigType.Controlnet_HrOption : ["dd", str, "hr_option" ],

    ConfigType.LoraEnv_CharacterDir : ["dir", str, "character_dir_path"],
    ConfigType.LoraEnv_StyleDir : ["dir", str, "style_dir_path"],
    ConfigType.LoraEnv_PoseDir : ["dir", str, "pose_dir_path"],
    ConfigType.LoraEnv_ItemDir : ["dir", str, "item_dir_path"],

    ConfigType.PresetTags_Prompt : ["str", str, "prompt"],
    ConfigType.PresetTags_NegPrompt : ["str", str, "negative_prompt"],
    ConfigType.PresetTags_IsFooter : ["bool", bool, "is_footer"],

    ConfigType.DefaultCheckPoint : ["dd", str, "default_checkpoint"],
    ConfigType.LoraEnv : ["dd", str, "lora_dir_env"],

    ConfigType.LoraGenerateTag_EnableCharacter : ["bool", bool, "enable_character"],
    ConfigType.LoraGenerateTag_EnableStyle : ["bool", bool, "enable_style"],
    ConfigType.LoraGenerateTag_EnablePose : ["bool", bool, "enable_pose"],
    ConfigType.LoraGenerateTag_EnableItem : ["bool", bool, "enable_item"],
    ConfigType.LoraGenerateTag_TagThCharacter : ["num", float, "tag_th_character"],
    ConfigType.LoraGenerateTag_TagThStyle : ["num", float, "tag_th_style"],
    ConfigType.LoraGenerateTag_TagThPose : ["num", float, "tag_th_pose"],
    ConfigType.LoraGenerateTag_TagThItem : ["num", float, "tag_th_item"],
    ConfigType.LoraGenerateTag_ProhibitedTagsCharacter : ["list", str_to_token, "prohibited_tags_character"],
    ConfigType.LoraGenerateTag_ProhibitedTagsStyle : ["list", str_to_token, "prohibited_tags_style"],
    ConfigType.LoraGenerateTag_ProhibitedTagsPose : ["list", str_to_token, "prohibited_tags_pose"],
    ConfigType.LoraGenerateTag_ProhibitedTagsItem : ["list", str_to_token, "prohibited_tags_item"],

    ConfigType.LoraBlockWeight_EnableLbwCharacter : ["bool", bool, "character_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetCharacter : ["str", str, "character_preset"],
    ConfigType.LoraBlockWeight_SssCharacter : ["dd", str, "character_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValueCharacter : ["num", int, "character_start_stop_step_value"],
    ConfigType.LoraBlockWeight_EnableLbwCharacter2 : ["bool", bool, "character2_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetCharacter2 : ["str", str, "character2_preset"],
    ConfigType.LoraBlockWeight_SssCharacter2 : ["dd", str, "character2_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValueCharacter2 : ["num", int, "character2_start_stop_step_value"],

    ConfigType.LoraBlockWeight_EnableLbwStyle : ["bool", bool, "style_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetStyle : ["str", str, "style_preset"],
    ConfigType.LoraBlockWeight_SssStyle : ["dd", str, "style_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValueStyle : ["num", int, "style_start_stop_step_value"],
    ConfigType.LoraBlockWeight_EnableLbwStyle2 : ["bool", bool, "style2_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetStyle2 : ["str", str, "style2_preset"],
    ConfigType.LoraBlockWeight_SssStyle2 : ["dd", str, "style2_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValueStyle2 : ["num", int, "style2_start_stop_step_value"],

    ConfigType.LoraBlockWeight_EnableLbwPose : ["bool", bool, "pose_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetPose : ["str", str, "pose_preset"],
    ConfigType.LoraBlockWeight_SssPose : ["dd", str, "pose_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValuePose : ["num", int, "pose_start_stop_step_value"],
    ConfigType.LoraBlockWeight_EnableLbwPose2 : ["bool", bool, "pose2_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetPose2 : ["str", str, "pose2_preset"],
    ConfigType.LoraBlockWeight_SssPose2 : ["dd", str, "pose2_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValuePose2 : ["num", int, "pose2_start_stop_step_value"],

    ConfigType.LoraBlockWeight_EnableLbwItem : ["bool", bool, "item_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetItem : ["str", str, "item_preset"],
    ConfigType.LoraBlockWeight_SssItem : ["dd", str, "item_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValueItem : ["num", int, "item_start_stop_step_value"],
    ConfigType.LoraBlockWeight_EnableLbwItem2 : ["bool", bool, "item2_enable_lbw"],
    ConfigType.LoraBlockWeight_PresetItem2 : ["str", str, "item2_preset"],
    ConfigType.LoraBlockWeight_SssItem2 : ["dd", str, "item2_start_stop_step"],
    ConfigType.LoraBlockWeight_SssValueItem2 : ["num", int, "item2_start_stop_step_value"],

    ConfigType.GenTxt2Img_EnableHr : ["bool", bool, "enable_hr"],
    ConfigType.GenTxt2Img_DenoisingStr : ["num", float, "denoising_strength"],
    ConfigType.GenTxt2Img_1pWidth : ["num", int, "firstphase_width"],
    ConfigType.GenTxt2Img_1pHeight : ["num", int, "firstphase_height"],
    ConfigType.GenTxt2Img_HrScale : ["num", float, "hr_scale"],
    ConfigType.GenTxt2Img_HrUpscaler : ["dd", str, "hr_upscaler"],
    ConfigType.GenTxt2Img_Hr2pSteps : ["num", int, "hr_second_pass_steps"],
    ConfigType.GenTxt2Img_HrResizeX : ["num", int, "hr_resize_x"],
    ConfigType.GenTxt2Img_HrResizeY : ["num", int, "hr_resize_y"],

    ConfigType.GenImg2Img_ResizeMode : ["dd", int, "resize_mode"],
    ConfigType.GenImg2Img_DenoisingStr : ["num", float, "denoising_strength"],
    ConfigType.GenImg2Img_ImgCfgScale : ["num", float, "image_cfg_scale"],
    ConfigType.GenImg2Img_MaskBlur : ["num", int, "mask_blur"],
    ConfigType.GenImg2Img_InpaintingFill : ["num", int, "inpainting_fill"],
    ConfigType.GenImg2Img_InpaintFullRes : ["bool", bool, "inpaint_full_res"],
    ConfigType.GenImg2Img_InpaintFullResPadding : ["num", int, "inpaint_full_res_padding"],
    ConfigType.GenImg2Img_InpaintingMaskInvert : ["num", int, "inpainting_mask_invert"],
    ConfigType.GenImg2Img_InitialNoiseMul : ["num", float, "initial_noise_multiplier"],

    ConfigType.CommonGen_SamplerName : ["dd", str, "sampler_name"],
    ConfigType.CommonGen_Scheduler : ["dd", str, "scheduler"],
    ConfigType.CommonGen_Steps : ["num", int, "steps"],
    ConfigType.CommonGen_CfgScale : ["num", float, "cfg_scale"],
    ConfigType.CommonGen_Width : ["num", int, "width"],
    ConfigType.CommonGen_Height : ["num", int, "height"],
    ConfigType.CommonGen_RestoreFaces : ["bool", bool, "restore_faces"],
    ConfigType.CommonGen_Tiling : ["bool", bool, "tiling"],
    ConfigType.CommonGen_DnSaveSamples : ["bool", bool, "do_not_save_samples"],
    ConfigType.CommonGen_DnSaveGrid : ["bool", bool, "do_not_save_grid"],
    ConfigType.CommonGen_NegPrompt : ["str", str, "negative_prompt"],
    ConfigType.CommonGen_Eta : ["num", float, "eta"],
    ConfigType.CommonGen_SendImages : ["bool", bool, "send_images"],
    ConfigType.CommonGen_SaveImages : ["bool", bool, "save_images"],

    ConfigType.PromptGen_ModelName : ["dd", str, "model_name"],
    ConfigType.PromptGen_Text : ["str", str, "text"],
    ConfigType.PromptGen_MinLength : ["num", int, "min_length"],
    ConfigType.PromptGen_MaxLength : ["num", int, "max_length"],
    ConfigType.PromptGen_NumBeams : ["num", int, "num_beams"],
    ConfigType.PromptGen_Temp : ["num", float, "temperature"],
    ConfigType.PromptGen_RepPenalty : ["num", float, "repetition_penalty"],
    ConfigType.PromptGen_LengthPref : ["num", float, "length_preference"],
    ConfigType.PromptGen_SamplingMode : ["dd", str, "sampling_mode"],
    ConfigType.PromptGen_TopK : ["num", int, "top_k"],
    ConfigType.PromptGen_TopP : ["num", float, "top_p"],

    ConfigType.OWGen_OWSteps : ["bool", bool, "overwrite_steps"],
    ConfigType.OWGen_OWSamplerName : ["bool", bool, "overwrite_sampler_name"],
    ConfigType.OWGen_OWScheduler : ["bool", bool, "overwrite_scheduler"],
    ConfigType.OWGen_OWCfgScale : ["bool", bool, "overwrite_cfg_scale"],
    ConfigType.OWGen_OWWidth : ["bool", bool, "overwrite_width"],
    ConfigType.OWGen_OWHeight : ["bool", bool, "overwrite_height"],
    ConfigType.OWGen_OWPrompt : ["bool", bool, "overwrite_prompt"],
    ConfigType.OWGen_OWNegPrompt : ["bool", bool, "overwrite_negative_prompt"],
    ConfigType.OWGen_OWSeed : ["bool", bool, "overwrite_seed"],
    ConfigType.OWGen_AddLora : ["bool", bool, "add_lora"],
    ConfigType.OWGen_AddPromptGen : ["bool", bool, "add_prompt_gen"],

    ConfigType.Adetailer_Model : ["dd", str, "1_ad_model"],
    ConfigType.Adetailer_Prompt : ["str", str, "1_ad_prompt"],
    ConfigType.Adetailer_NegPrompt : ["str", str, "1_ad_negative_prompt"],
    ConfigType.Adetailer_Model2 : ["dd", str, "2_ad_model"],
    ConfigType.Adetailer_Prompt2 : ["str", str, "2_ad_prompt"],
    ConfigType.Adetailer_NegPrompt2 : ["str", str, "2_ad_negative_prompt"],
    ConfigType.Adetailer_Model3 : ["dd", str, "3_ad_model"],
    ConfigType.Adetailer_Prompt3 : ["str", str, "3_ad_prompt"],
    ConfigType.Adetailer_NegPrompt3 : ["str", str, "3_ad_negative_prompt"],

    ConfigType.Sam_SamModelName : ["dd", str, "sam_model_name"],
    ConfigType.Sam_DinoModelName : ["str", str, "dino_model_name"],



    ConfigType.Input_Prompt_CharacterLora : ["lora", str, "character_lora"],
    ConfigType.Input_Prompt_CharacterLora2 : ["lora", str, "character_lora2"],
    ConfigType.Input_Prompt_StyleLora : ["lora", str, "style_lora"],
    ConfigType.Input_Prompt_StyleLora2 : ["lora", str, "style_lora2"],
    ConfigType.Input_Prompt_PoseLora : ["lora", str, "pose_lora"],
    ConfigType.Input_Prompt_PoseLora2 : ["lora", str, "pose_lora2"],
    ConfigType.Input_Prompt_ItemLora : ["lora", str, "item_lora"],
    ConfigType.Input_Prompt_ItemLora2 : ["lora", str, "item_lora2"],
    ConfigType.Input_Prompt_PresetTags : ["tags", str, "preset_tags"],
    ConfigType.Input_Prompt_Header : ["large_str", str, "header"],
    ConfigType.Input_Prompt_Footer : ["large_str", str, "footer"],

    ConfigType.Input_CheckPoint : ["dd", str, "checkpoint"],
    ConfigType.Input_Seed : ["dd", str, "seed"],

    ConfigType.Input_OW_PngInfo : ["image2", str, "png_info"],

    ConfigType.Input_PromptGen_Type : ["dd", str, "type"],
    ConfigType.Input_PromptGen_IsFooter : ["bool", bool, "is_footer"],

    ConfigType.Input_Seq_Type : ["dd", str, "type"],
    ConfigType.Input_Seq_InputImage : ["image", str, "input_image"],
    ConfigType.Input_Seq_OutputScale : ["num", float, "output_scale"],
    ConfigType.Input_Seq_ControlnetType : ["dd", str, "type"],
    ConfigType.Input_Seq_ControlnetImage : ["image", str, "image"],
    ConfigType.Input_Seq_ControlnetCNTarget : ["cn_target", str, "cn_target"],

}

class ConfigItem:
    def __init__(self, name:str, val, ui_type, options, format_func, page=None):
        self.name = name
        self.org_val = val
        self.val = val
        self.ui_type = ui_type
        self.options = options
        self.format_func = format_func
        self.page : ft.Page = page
    
    def _create_picker_control(self, picker_type, on_pick_ok):
        # "dir", "png", "mp4"
        def on_click(e: ft.ControlEvent):
            def on_pick(e: ft.FilePickerResultEvent):
                if picker_type == "dir":
                    if e.path:
                        on_pick_ok(Path(e.path).as_posix())
                else:
                    if e.files:
                        on_pick_ok(Path(e.files[0].path).as_posix())

                self.page.overlay.remove(pick_files_dialog)
                self.page.update()

            pick_files_dialog = ft.FilePicker(on_result=on_pick)
            self.page.overlay.append(pick_files_dialog)
            self.page.update()
            if picker_type == "dir":
                pick_files_dialog.get_directory_path( f"Select Directory ({self.name})" )
            elif picker_type == "png":
                pick_files_dialog.pick_files(
                    dialog_title=f"Select Png File ({self.name})",
                    allow_multiple=False,
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions = ["png","PNG"]
                )
            elif picker_type == "mp4":
                pick_files_dialog.pick_files(
                    dialog_title=f"Select Mp4 File ({self.name})",
                    allow_multiple=False,
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions = ["mp4","MP4"]
                )

        btn = ft.IconButton(icon=ft.icons.FOLDER, on_click=on_click)

        return btn
    
    def _create_dir_control(self):

        big_label_style= ft.TextStyle( size=20 )

        row = ft.Row(controls=[],alignment= ft.MainAxisAlignment.CENTER)
        controls = []

        def on_pick_ok(v):
            txt.value = v
            self.val = self.format_func(txt.value)

        def on_change(e: ft.ControlEvent):
            val = e.control.value
            self.val = self.format_func(val)

        btn = self._create_picker_control("dir", on_pick_ok)
        controls.append(btn)
        txt = ft.TextField(self.val,label=self.name, label_style=big_label_style, expand=True, dense=True, on_change=on_change, text_size=15)
        controls.append(txt)

        row.controls = controls
        return row

    def _create_image_control(self, ui_type):

        def update_value(v):
            if ui_type == "image":
                self.val[0] = v if type(v)==list else [v]
            else:
                self.val = v

        if ui_type == "image":
            image_rule, focus_target, focus_scale = validate_focus_rule(self.val)
            self.val = ["", focus_target, focus_scale]
            update_value(image_rule)
        else:
            image_rule = self.val

        input_type, rule = validate_image_rule(image_rule)

        big_label_style= ft.TextStyle( size=20 )

        row = ft.Row(controls=[],alignment= ft.MainAxisAlignment.CENTER)


        def create_control(input_type, rule):
            controls = [label, type_cont]
            if input_type == "image_file":
                def on_pick_ok_image_file(v):
                    text.value = v
                    update_value(text.value)

                def on_change_image_file(e: ft.ControlEvent):
                    update_value(text.value)

                btn = self._create_picker_control("png", on_pick_ok_image_file)
                controls.append(btn)
                text = ft.TextField(rule, label="input image_file path", expand=True, on_change=on_change_image_file, text_size=15)
                controls.append(text)

            elif input_type == "image_directory":
                def on_pick_ok_image_dir(v):
                    text.value = v
                    update_value([text.value, dd1.value, dd2.value])

                def on_change_image_directory(e: ft.ControlEvent):
                    update_value([text.value, dd1.value, dd2.value])
                
                btn = self._create_picker_control("dir", on_pick_ok_image_dir)
                controls.append(btn)
                text = ft.TextField(rule[0], label="input image_directory path", expand=True, on_change=on_change_image_directory, text_size=15)
                controls.append(text)

                opt = [ft.dropdown.Option(o) for o in ["random","sort"]]
                dd1 = ft.Dropdown(rule[1],label="pick rule", width=150, options=opt, on_change=on_change_image_directory, options_fill_horizontally=True, text_size=15)
                controls.append(dd1)

                opt = [ft.dropdown.Option(o) for o in ["@random","@random_per_seq","@random_once"]]
                dd2 = ft.Dropdown(rule[2],label="pick timing", width=150, options=opt, on_change=on_change_image_directory, options_fill_horizontally=True, text_size=15)
                controls.append(dd2)

            elif input_type == "movie_file":
                def on_pick_ok_movie_file(v):
                    text.value = v
                    update_value([text.value, float(num.value)])

                def on_change_movie_file(e: ft.ControlEvent):
                    update_value([text.value, float(num.value)])
                    
                btn = self._create_picker_control("mp4", on_pick_ok_movie_file)
                controls.append(btn)
                text = ft.TextField(rule[0], label="input movie_file path", expand=True, on_change=on_change_movie_file, text_size=15)
                controls.append(text)
                num = ft.TextField(rule[1], label="input interval seconds", input_filter=ft.InputFilter(regex_string=r"^(\d+(\.\d*)?|\.\d+)$", allow=True), on_change=on_change_movie_file, text_size=15)
                controls.append(num)

            elif input_type == "generation_result":
                def on_change_generation_result(e: ft.ControlEvent):
                    update_value(int(num.value))
                num = ft.TextField(rule, label="input generation number", expand=True, input_filter=ft.InputFilter(regex_string=r"^(\d+)$", allow=True), on_change=on_change_generation_result, text_size=15)
                controls.append(num)

            if ui_type == "image":
                controls.append(f_text)
                controls.append(f_num)

            return controls

        default_value = {
            "image_file" : "",
            "image_directory" : ["","random","@random"],
            "movie_file" : ["", 1.0],
            "generation_result" : 0,
        }

        label = ft.Text(self.name, width=100)

        def on_change(e: ft.ControlEvent):
            input_type = type_cont.value
            update_value(default_value[input_type])
            row.controls = create_control(input_type, default_value[input_type])
            self.page.update()

        if ui_type == "image":
            opt = [ft.dropdown.Option(o) for o in ["image_file","image_directory","movie_file","generation_result"]]
        else:
            opt = [ft.dropdown.Option(o) for o in ["image_file","image_directory"]]

        type_cont = ft.Dropdown(input_type,label="input_type", label_style=big_label_style, width=150, options=opt, on_change=on_change, options_fill_horizontally=True, text_size=15)

        if ui_type == "image":
            def on_change_focus(e: ft.ControlEvent):
                self.val[1] = f_text.value
                self.val[2] = f_num.value

            f_text = ft.TextField(focus_target, label="focus target (ex. face)", width=200, on_change=on_change_focus, text_size=15)
            f_num = ft.TextField(focus_scale, label="input focus scale", width=100, input_filter=ft.InputFilter(regex_string=r"^(\d+(\.\d*)?|\.\d+)$", allow=True), on_change=on_change_focus, text_size=15)

        row.controls = create_control(input_type, rule)

        return row
        
    def _create_lora_selector(self, on_click_func):
        conf2type={
            "character_lora":LoraType.Character,
            "character_lora2":LoraType.Character,
            "style_lora":LoraType.Style,
            "style_lora2":LoraType.Style,
            "pose_lora":LoraType.Pose,
            "pose_lora2":LoraType.Pose,
            "item_lora":LoraType.Item,
            "item_lora2":LoraType.Item,
        }


        def get_lora_list(lora_type:LoraType):
            from sd_batch_runner.lora import get_lora_files_and_thumbs
            lora_list = get_lora_files_and_thumbs(lora_type, (200,200))

            lora_controls = []

            def create_item(lora, thumb):
                controls = [ ]
                if thumb:
                    controls.append( ft.Image(src_base64=thumb) )
                controls.append( ft.ElevatedButton(text=lora, width=200, on_click=on_click_func, data=lora) )
                return ft.Column( controls, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER )
            
            for lora, thumb in lora_list:
                lora_controls.append( ft.Container(create_item(lora, thumb), border=ft.border.all(5, ft.colors.BLACK45) ) )

            return lora_controls, [l[0] for l in lora_list]

        def _get_lora_list(lora_type:LoraType):
            from sd_batch_runner.lora import get_lora_files_and_preview_paths
            lora_list = get_lora_files_and_preview_paths(lora_type)

            lora_controls = []

            def create_item(lora, thumb):
                controls = [ ]
                if thumb:
                    controls.append( ft.Image(src=thumb, width=200, height=200) )
                controls.append( ft.ElevatedButton(text=lora, width=200, on_click=on_click_func, data=lora) )
                return ft.Column( controls, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER )
            
            for lora, thumb in lora_list:
                lora_controls.append( ft.Container(create_item(lora, thumb), border=ft.border.all(5, ft.colors.BLACK45) ) )

            return lora_controls, [l[0] for l in lora_list]

        dialog = ft.AlertDialog(
            title=ft.Text(f"Select {self.name}"),
        )

        lora_list, lora_name_list = get_lora_list( conf2type[self.name] )
        lora_dict = {}
        for lc, ln in zip(lora_list, lora_name_list):
            lora_dict[ln] = lc

        visible_lora_name_list = lora_name_list.copy()

        def on_filter_change(e: ft.ControlEvent):
            nonlocal visible_lora_name_list
            filter = filter_text.value
            filter = filter.lower()
            filter_list = filter.split("|")
            filter_list = [f for f in filter_list if f]

            if len(filter_list) == 0:
                visible_lora_name_list = lora_name_list.copy()
            else:
                result = []
                for f in filter_list:
                    result += [name for name in lora_name_list if name.lower().find(f) != -1]

                visible_lora_name_list = list(dict.fromkeys(result))


            lora_col.controls = show_lora_list()
            self.page.update()


        filter_text = ft.TextField("", label="filter", on_change=on_filter_change)

        def show_lora_list():
            lora_controls = []
            tmp =[]

            visible_lora_list = [ lora_dict[n] for n in visible_lora_name_list]

            for l in visible_lora_list:
                tmp.append(l)
                if len(tmp) == 5:
                    lora_controls.append(
                        ft.Row(tmp, vertical_alignment= ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER)
                    )
                    tmp = []
            
            if tmp:
                lora_controls.append(
                    ft.Row(tmp, vertical_alignment= ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER)
                )
            return lora_controls
        

        lora_col = ft.Column(
            controls=show_lora_list(),
            scroll=ft.ScrollMode.ALWAYS,
            expand=True
        )

        col = ft.Column(
            controls=[filter_text, lora_col],
            expand=True
        )

        dialog.content = col

        return dialog

    def _create_lora_control(self):

        random_opt = [
            "@random", "@random_per_seq", "@random_once"
        ]

        def validate_lora_rule(lora_rule):
            # [random_keyword, [ lora_str, filter ]]
            # [single_file, lora_str]
            if type(lora_rule) == list:
                pass
            else:
                lora_rule = [lora_rule, 1.0]
            
            if lora_rule[0] in random_opt:
                if type(lora_rule[1]) == list:
                    pass
                else:
                    lora_rule = [lora_rule[0], [lora_rule[1], ""]]
            
            if lora_rule[0] in random_opt:
                return "random", lora_rule
            else:
                return "single_file", lora_rule


        input_type, rule = validate_lora_rule(self.val)

        big_label_style= ft.TextStyle( size=20 )

        row = ft.Row(controls=[],alignment= ft.MainAxisAlignment.START)

        def create_control(input_type, rule):
            #logger.info(f"{input_type=} {rule=}")

            controls = [label,type_cont]
            if input_type == "random":

                def on_change_random(e: ft.ControlEvent):
                    self.val = [dd1.value, [float(num.value), filter_text.value]]

                opt = [ft.dropdown.Option(o) for o in ["@random","@random_per_seq","@random_once"]]
                dd1 = ft.Dropdown(rule[0],label="pick timing", options=opt, width=200, on_change=on_change_random, options_fill_horizontally=True, text_size=15)
                controls.append(dd1)
                filter_text = ft.TextField(rule[1][1], label="filter (ex. asuka|ayanami)", expand=True, on_change=on_change_random, text_size=15)
                controls.append(filter_text)
                num = ft.TextField(rule[1][0], label="input lora strength", width=150, input_filter=ft.InputFilter(regex_string=r"^(\d+(\.\d*)?|\.\d+)$", allow=True), on_change=on_change_random, text_size=15)
                controls.append(num)

            elif input_type == "single_file":

                lora_selector = None

                def on_lora_selected(e: ft.ControlEvent):
                    nonlocal lora_selector
                    text.value = e.control.text + ".safetensors"
                    self.val = [text.value, float(num.value)]
                    self.page.close(lora_selector)
                    self.page.update()

                
                def on_icon_click(e: ft.ControlEvent):
                    nonlocal lora_selector
                    lora_selector = self._create_lora_selector( on_lora_selected )
                    self.page.open(lora_selector)
                    self.page.update()


                btn = ft.IconButton(icon=ft.icons.FOLDER_OPEN, width=50, on_click=on_icon_click)
                controls.append(btn)

                def on_change_single(e: ft.ControlEvent):
                    self.val = [text.value, float(num.value)]

                text = ft.TextField(rule[0], label="lora file name", expand=True, on_change=on_change_single, text_size=15)
                controls.append(text)
                num = ft.TextField(rule[1], label="input lora strength", width=150, input_filter=ft.InputFilter(regex_string=r"^(\d+(\.\d*)?|\.\d+)$", allow=True), on_change=on_change_single, text_size=15)
                controls.append(num)


            return controls


        default_value = {
            "random" : ["@random", [1.0,""]],
            "single_file" : ["", 1.0],
        }

        label = ft.Text(self.name, width=100)


        def on_change(e: ft.ControlEvent):
            input_type = type_cont.value
            self.val = default_value[input_type]
            row.controls = create_control(input_type, self.val)
            self.page.update()

        opt = [ft.dropdown.Option(o) for o in ["random","single_file"]]
        type_cont = ft.Dropdown(input_type,label="input_type", label_style=big_label_style, width=150, options=opt, on_change=on_change, options_fill_horizontally=True, text_size=15)

        row.controls = create_control(input_type, rule)

        return row

    def _create_cn_target_control(self):

        def validate_rule(rule):
            if type(rule) == int:
                return "use_cache", rule
            else:
                return "specify_target", rule

        input_type, rule = validate_rule(self.val)

        big_label_style= ft.TextStyle( size=20 )

        row = ft.Row(controls=[],alignment= ft.MainAxisAlignment.START)

        def create_control(input_type, rule):
            controls = [label,type_cont]
            if input_type == "specify_target":

                def on_change_specify_target(e: ft.ControlEvent):
                    self.val = text.value

                text = ft.TextField(rule, label="Specify target of controlnet (ex. a girl)", expand=True, on_change=on_change_specify_target, text_size=15)
                controls.append(text)

            elif input_type == "use_cache":

                def on_change_use_cache(e: ft.ControlEvent):
                    self.val = int(num.value)

                num = ft.TextField(rule, label="input generation number", expand=True, input_filter=ft.InputFilter(regex_string=r"^(\d+)$", allow=True), on_change=on_change_use_cache, text_size=15)
                controls.append(num)

            return controls

        default_value = {
            "specify_target" : "",
            "use_cache" : 0,
        }

        label = ft.Text(self.name, width=100)

        def on_change(e: ft.ControlEvent):
            input_type = type_cont.value
            self.val = default_value[input_type]
            row.controls = create_control(input_type, self.val)
            self.page.update()

        opt = [ft.dropdown.Option(o) for o in ["specify_target","use_cache"]]
        type_cont = ft.Dropdown(input_type, label="input_type", label_style=big_label_style, width=200, options=opt, on_change=on_change, options_fill_horizontally=True, text_size=15)

        row.controls = create_control(input_type, rule)

        return row

    def _create_tags_control(self):
        big_label_style= ft.TextStyle( size=20 )

        preset_tags_list = get_preset_tags_list()

        def on_change(e: ft.ControlEvent):
            duplicate = False
            for preset in self.val:
                if preset == tags_dd.value:
                    duplicate = True
            if duplicate:
                return
            
            self.val = self.val + [tags_dd.value]

            row.controls = create_controls()
            self.page.update()

        def on_click(e: ft.ControlEvent):
            self.val.remove(e.control.label.value)
            row.controls = create_controls()
            self.page.update()
        
        def create_controls():
            controls=[]
            for preset in self.val:
                chip = ft.Chip(ft.Text(preset), leading=ft.Icon(ft.icons.DELETE), on_click=on_click, col=3)
                controls.append(chip)
            tags_row = ft.ResponsiveRow(controls=controls,alignment= ft.MainAxisAlignment.START, expand=True)
            tags = ft.Container(
                content=tags_row,
                bgcolor=ft.colors.BLACK12,
                expand=True,
                margin=5,
                padding=10,
                border_radius=5,

            )
            return [tags_dd, tags]

        opt = [ft.dropdown.Option(o) for o in preset_tags_list]
        tags_dd = ft.Dropdown(preset_tags_list[0], label="preset_tags", label_style=big_label_style, on_change=on_change, width=200, options=opt, options_fill_horizontally=True, text_size=15)

        row = ft.Row(controls=[],alignment= ft.MainAxisAlignment.START)
        row.controls=create_controls()

        return row

    
    def create_ui_control(self):
        def on_change(e: ft.ControlEvent):
            val = e.control.value
            self.val = self.format_func(val)

        big_label_style= ft.TextStyle( size=20 )
        if self.ui_type == "dd":
            opt = [ft.dropdown.Option(o) for o in self.options]
            cont = ft.Dropdown(self.val,label=self.name, label_style=big_label_style, expand=True, dense=True, options=opt, on_change=on_change, options_fill_horizontally=True, text_size=15)
        elif self.ui_type == "num":
            cont = ft.TextField(self.val,label=self.name, label_style=big_label_style, expand=True, dense=True, input_filter=ft.InputFilter(regex_string=r"^-?(\d+(\.\d*)?|\.\d+)$", allow=True), on_change=on_change, text_size=15)
        elif self.ui_type == "list":
            cont = ft.TextField(",".join(self.val),label=self.name, label_style=big_label_style, expand=True, dense=True, on_change=on_change, text_size=15)
        elif self.ui_type == "str":
            cont = ft.TextField(self.val,label=self.name, label_style=big_label_style, expand=True, dense=True, on_change=on_change, text_size=15)
        elif self.ui_type == "large_str":
            cont = ft.TextField(self.val,label=self.name, label_style=big_label_style, expand=True, dense=True, on_change=on_change, text_size=15, multiline=True, min_lines=4, height=100)
        elif self.ui_type == "bool":
            cont = ft.Checkbox(value=self.val,label=self.name, expand=True, on_change=on_change)
        elif self.ui_type == "image":
            cont = self._create_image_control("image")
        elif self.ui_type == "image2":
            cont = self._create_image_control("image2")
        elif self.ui_type == "lora":
            cont = self._create_lora_control()
        elif self.ui_type == "dir":
            cont = self._create_dir_control()
        elif self.ui_type == "cn_target":
            cont = self._create_cn_target_control()
        elif self.ui_type == "tags":
            cont = self._create_tags_control()

        return cont
    
    def is_update(self):
        return self.val != self.org_val
    
    def notify_updated(self):
        self.org_val = self.val

    def validate(self):
        if "image":
            # [ [] , "" , 1.0 ]
            if type(self.val) == list:
                if type(self.val[0]) == list:
                    if self.val[0][0] == "":
                        self.val = ""

    @classmethod
    def create(cls, config_type:ConfigType, org_val, options=None, page=None):

        ui_type, format_func, name = CONFIG_MAP[config_type]

        return cls(name, org_val, ui_type, options, format_func, page)



class ViewCreator():
    def __init__(self, page, route, title, json_path, parent_route="/view_top"):
        self.page = page
        self.route = route
        self.title = title
        self.parent_route = parent_route

        self.json_path = Path(json_path)

        self.load()
        
        self.config_items = {}
        self.config_list = []

    def load(self):
        self.info = {}
        if self.json_path.is_file():
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.info = json.load(f)
        else:
            logger.info(f"{self.json_path} not found")
    
    def save(self):
        json_text = json.dumps(self.info, indent=4, ensure_ascii=False)
        self.json_path.write_text(json_text, encoding="utf-8")


    def back_to_top(self):
        if self.is_update() == False:
            self.page.go(self.parent_route)
        else:
            def on_click(e: ft.ControlEvent):
                if e.control.text == "Yes":
                    self.page.go(self.parent_route)
                self.page.close(move_dlg_modal)

            move_dlg_modal = ft.AlertDialog(
                modal=True,
                title=ft.Text("Some unsaved data exists."),
                content=ft.Text("Do you wish to move?"),
                actions=[
                    ft.TextButton("Yes", on_click=on_click),
                    ft.TextButton("No", on_click=on_click)
                ],
                actions_alignment=ft.MainAxisAlignment.END,
                on_dismiss=lambda e: self.page.add(
                    ft.Text("Modal dialog dismissed"),
                ),
            )
            self.page.open(move_dlg_modal)

    def get_opts(self, t : ConfigType):
        opt = DD_OPTION_MAP.get(t, None)
        return opt

    def create_config_item(self, c, key=None):
        if key:
            c = self.info[key]

        tmp = {}
        for t in self.config_list:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)

            item = ConfigItem.create( t, c[name], opt, self.page)
            tmp[name] = item
        return tmp


    def load_config(self):
        for key in self.info:
            self.config_items[key] = self.create_config_item(None, key)

    def save_config(self):
        self.validate_value()

        new_info = {}
        for key in self.config_items:
            c = self.config_items[key]
            for name in c:
                item : ConfigItem = c[name]
                if key not in new_info:
                    new_info[key] = {}
                new_info[key][name] = item.val
        self.info = new_info
        
    
    def is_update(self):
        for key in self.config_items:
            c = self.config_items[key]
            for name in c:
                item : ConfigItem = c[name]
                if item.is_update():
                    return True
        
        if len(self.info.keys() ^ self.config_items.keys()) != 0:
            return True

        return False
    
    def notify_updated(self):
        def inner_notify(c):
            if type(c) == dict:
                for name in c:
                    inner_notify(c[name])
            elif type[c] == list:
                for cc in c:
                    inner_notify(cc)
            else:
                c.notify_updated()
        
        inner_notify(self.config_items)

    def validate_value(self):
        pass


    def show_item(self, key):

        exp = ft.ExpansionPanel(
            bgcolor=ft.colors.GREEN_50,
            can_tap_header=True,
            header=ft.ListTile(title=ft.Text(f"{key}")),
            data=key
        )

        c = self.config_items[key]

        controls = [ c[name].create_ui_control() for name in c ]

        exp.content = ft.Container(
            ft.Column(
                controls=controls,
                horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                tight=True
            ),
            margin=0,
            padding=10,
            border_radius=0,
        )

        self.panel.controls.append(exp)


    def warn_save(self):
        def handle_close(e):
            self.page.close(save_dlg_modal)

        def save(e: ft.ControlEvent):
            self.save_config()
            self.save()
            self.notify_updated()

            self.page.close(save_dlg_modal)
        
        save_dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Text("Please confirm"),
            content=ft.Text("Do you really want to Save?"),
            actions=[
                ft.TextButton("Yes", on_click=save),
                ft.TextButton("No", on_click=handle_close)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: self.page.add(
                ft.Text("Modal dialog dismissed"),
            ),
        )
        self.page.open(save_dlg_modal)

    def create_tail_icon(self):
        def warn_save(e: ft.ControlEvent):
            self.warn_save()

        return ft.Row( controls=[
            ft.IconButton(icon=ft.icons.SAVE, width=200, on_click=warn_save),
            ft.Text( self.json_path.absolute() ),
        ] )

    def create(self):
        def back_to_top(e: ft.ControlEvent):
            self.back_to_top()

        view = ft.View(self.route, [
            ft.AppBar(
                leading=ft.IconButton(icon=ft.icons.ARROW_BACK, width=200, on_click=back_to_top),
                title=ft.Text(self.title),
                bgcolor=ft.colors.BLUE),
        ])

        self.load_config()

        self.panel = ft.ExpansionPanelList(
            expand_icon_color=ft.colors.BLACK,
            elevation=8,
            divider_color=ft.colors.BLACK,
            controls=[
            ],
        )

        for key in self.config_items:
            self.show_item(key)

        col = ft.Column(
            controls=[self.panel],
            scroll=ft.ScrollMode.ALWAYS,
            expand=True,
            tight=True
        )
        
        view.controls.append(col)

        view.controls.append( ft.Container(
            content=self.create_tail_icon(),
            bgcolor=ft.colors.BLUE_50
        ))

        return view


class TabViewCreator(ViewCreator):

    def create_config_item(self, c, key=None):
        c = self.info[key]

        tmp = {}
        for t in self.config_list[key]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)

            item = ConfigItem.create( t, c[name], opt, self.page)
            tmp[name] = item
        return tmp
    
    def show_tab(self, key):

        c = self.config_items[key]
        controls = [ c[name].create_ui_control() for name in c ]

        tab = ft.Tab(
            text=key,
            content = 
            ft.Container(
                ft.Column(
                    controls=controls,
                    horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                    scroll=ft.ScrollMode.ALWAYS,
                    expand=True,
                    adaptive=True,
                    tight=True
                ),
                padding=10,
                bgcolor= ft.colors.GREEN_50,
            )
            ,
            visible=True,
            adaptive=True
        )


        self.tab_list.tabs.append(tab)


    def show_tabs(self):
        for key in self.tab_order:
            self.show_tab(key)


    def create(self):
        def back_to_top(e: ft.ControlEvent):
            self.back_to_top()

        view = ft.View(self.route, [
            ft.AppBar(
                leading=ft.IconButton(icon=ft.icons.ARROW_BACK, width=200, on_click=back_to_top),
                title=ft.Text(self.title),
                bgcolor=ft.colors.BLUE),
        ])

        self.load_config()

        self.tab_list = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            expand=True,
            divider_color=ft.colors.BLACK,
            indicator_color=ft.colors.BLACK,
        )

        self.show_tabs()

        view.controls.append(self.tab_list)

        view.controls.append( ft.Container(
            content=self.create_tail_icon(),
            bgcolor=ft.colors.BLUE_50
        ))

        return view


class ControlnetViewCreator(ViewCreator):
    def __init__(self, page):
        super().__init__(page, "/view_controlnet", "Controlnet Alias Setting", "controlnet.json")

        self.controlnet_model_list ,self.controlnet_module_list = get_controlnet_list()

        self.config_list = [
            ConfigType.Controlnet_Module,
            ConfigType.Controlnet_Model,
            ConfigType.Controlnet_Weight,
            ConfigType.Controlnet_ResizeMode,
            ConfigType.Controlnet_LowVram,
            ConfigType.Controlnet_ProcessorRes,
            ConfigType.Controlnet_Tha,
            ConfigType.Controlnet_Thb,
            ConfigType.Controlnet_GuidanceStart,
            ConfigType.Controlnet_GuidanceEnd,
            ConfigType.Controlnet_ControlMode,
            ConfigType.Controlnet_PixelPerfect,
            ConfigType.Controlnet_HrOption,        
        ]

   
    def get_opts(self, t : ConfigType):
        if t == ConfigType.Controlnet_Module:
            opt = self.controlnet_module_list
        elif t == ConfigType.Controlnet_Model:
            opt = self.controlnet_model_list
        else:
            opt = DD_OPTION_MAP.get(t, None)
        return opt

    def show_item(self, key):

        def create_delete(exp):
            def handle_delete(e: ft.ControlEvent):
                name = e.control.data.data
                self.panel.controls.remove(e.control.data)
                self.config_items.pop(name)
                self.page.update()
            return ft.IconButton(icon=ft.icons.DELETE, width=200, on_click=handle_delete, data=exp)

        exp = ft.ExpansionPanel(
            bgcolor=ft.colors.GREEN_50,
            can_tap_header=True,
            header=ft.ListTile(title=ft.Text(f"{key}")),
            data=key
        )

        c = self.config_items[key]

        controls = [ c[name].create_ui_control() for name in c ]

        controls.append( ft.Row( controls=[
            create_delete(exp),
            ],
            alignment= ft.MainAxisAlignment.CENTER
        ))
        
        exp.content = ft.Container(
            ft.Column(
                controls=controls,
                horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                tight=True,
            ),
            margin=0,
            padding=10,
            border_radius=0,
        )

        self.panel.controls.append(exp)

    def append_item(self):
        def handle_close(e: ft.ControlEvent):
            self.page.close(append_dlg)

        def append_new_item(e: ft.ControlEvent):
            new_key = new_name_txt.value
            if not new_key:
                return
            self.page.close(append_dlg)
            if new_key not in self.config_items.keys():
                new_item = {
                    "module": "none",
                    "model": self.controlnet_model_list[1],
                    "weight": 0.5,
                    "resize_mode": "Resize and Fill",
                    "low_vram": False,
                    "processor_res": 512,
                    "threshold_a": 64,
                    "threshold_b": 64,
                    "guidance_start": 0.0,
                    "guidance_end": 0.5,
                    "control_mode": 0,
                    "pixel_perfect": True,
                    "hr_option": "Both"
                }

                self.config_items[new_key] = self.create_config_item(new_item)
                
                self.show_item(new_key)
                self.page.update()

        new_name_txt = ft.TextField("",label="Enter new item name")

        append_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Create New Controlnet Alias"),
            content=new_name_txt,
            actions=[
                ft.TextButton("Yes", on_click=append_new_item),
                ft.TextButton("No", on_click=handle_close)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: self.page.add(
                ft.Text("Modal dialog dismissed"),
            ),
        )
        self.page.open(append_dlg)

    def create_tail_icon(self):
        def append_item(e: ft.ControlEvent):
            self.append_item()
        def warn_save(e: ft.ControlEvent):
            self.warn_save()

        return ft.Row( controls=[
            ft.IconButton(icon=ft.icons.ADD, width=200, on_click=append_item),
            ft.IconButton(icon=ft.icons.SAVE, width=200, on_click=warn_save),
            ft.Text( self.json_path.absolute() ),
        ] )


class LoraEnvViewCreator(ViewCreator):
    def __init__(self, page):
        super().__init__(page, "/view_lora", "Lora Directory Setting", "lora_dir.json")

        self.config_list = [
            ConfigType.LoraEnv_CharacterDir,
            ConfigType.LoraEnv_StyleDir,
            ConfigType.LoraEnv_PoseDir,
            ConfigType.LoraEnv_ItemDir,
        ]

    def append_item(self):
        def handle_close(e: ft.ControlEvent):
            self.page.close(append_dlg)

        def append_new_item(e: ft.ControlEvent):
            new_key = new_name_txt.value
            if not new_key:
                return
            self.page.close(append_dlg)
            if new_key not in self.config_items.keys():
                new_item = {
                    "character_dir_path": "",
                    "style_dir_path": "",
                    "pose_dir_path": "",
                    "item_dir_path": "",
                }

                self.config_items[new_key] = self.create_config_item(new_item)
                
                self.show_item(new_key)
                self.page.update()

        new_name_txt = ft.TextField("",label="Enter new item name")

        append_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Create New Lora Env"),
            content=new_name_txt,
            actions=[
                ft.TextButton("Yes", on_click=append_new_item),
                ft.TextButton("No", on_click=handle_close)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: self.page.add(
                ft.Text("Modal dialog dismissed"),
            ),
        )
        self.page.open(append_dlg)

    def create_tail_icon(self):
        def append_item(e: ft.ControlEvent):
            self.append_item()
        def warn_save(e: ft.ControlEvent):
            self.warn_save()

        return ft.Row( controls=[
            ft.IconButton(icon=ft.icons.ADD, width=200, on_click=append_item),
            ft.IconButton(icon=ft.icons.SAVE, width=200, on_click=warn_save),
            ft.Text( self.json_path.absolute() ),
        ] )

    def show_item(self, key):

        def create_delete(exp):
            def handle_delete(e: ft.ControlEvent):
                name = e.control.data.data
                self.panel.controls.remove(e.control.data)
                self.config_items.pop(name)
                self.page.update()
            return ft.IconButton(icon=ft.icons.DELETE, width=200, on_click=handle_delete, data=exp)

        exp = ft.ExpansionPanel(
            bgcolor=ft.colors.GREEN_50,
            can_tap_header=True,
            header=ft.ListTile(title=ft.Text(f"{key}")),
            data=key,
        )

        c = self.config_items[key]

        controls = [ c[name].create_ui_control() for name in c ]

        controls.append( ft.Row( controls=[
            create_delete(exp),
            ],
            alignment= ft.MainAxisAlignment.CENTER,
            tight=True,
        ))
        
        exp.content = ft.Container(
            ft.Column(
                controls=controls,
                horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                tight=True,
            ),
            margin=0,
            padding=10,
            border_radius=0,
        )

        self.panel.controls.append(exp)


class PresetTagsViewCreator(LoraEnvViewCreator):
    def __init__(self, page):
        super(LoraEnvViewCreator, self).__init__(page, "/view_preset_tags", "Preset Tags Setting", "preset_tags.json")

        self.config_list = [
            ConfigType.PresetTags_Prompt,
            ConfigType.PresetTags_NegPrompt,
            ConfigType.PresetTags_IsFooter,
        ]

    def append_item(self):
        def handle_close(e: ft.ControlEvent):
            self.page.close(append_dlg)

        def append_new_item(e: ft.ControlEvent):
            new_key = new_name_txt.value
            if not new_key:
                return
            self.page.close(append_dlg)
            if new_key not in self.config_items.keys():
                new_item = {
                    "prompt": "",
                    "negative_prompt": "",
                    "is_footer": True,
                }

                self.config_items[new_key] = self.create_config_item(new_item)
                
                self.show_item(new_key)
                self.page.update()

        new_name_txt = ft.TextField("",label="Enter new item name")

        append_dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Create New Preset Tag"),
            content=new_name_txt,
            actions=[
                ft.TextButton("Yes", on_click=append_new_item),
                ft.TextButton("No", on_click=handle_close)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: self.page.add(
                ft.Text("Modal dialog dismissed"),
            ),
        )
        self.page.open(append_dlg)



class ConfigViewCreator(TabViewCreator):
    def __init__(self, page):
        super().__init__(page, "/view_config", "Config", "config.json")

        self.checkpoint_list = get_checkpoint_list()
        self.lora_dir_env_list = get_lora_dir_env_list()
        self.latent_upscale_mode_list = get_latent_upscale_mode_list()
        self.sampler_list = get_sampler_list()
        self.scheduler_list = get_scheduler_list()
        self.sam_model_list = get_sam_model_list()

        self.tab_order = [
            "main",
            "lora_generate_tag",
            "lora_block_weight",
            "generation_setting_txt2img",
            "generation_setting_img2img",
            "generation_setting_common",
            "prompt_gen_setting",
            "overwrite_generation_setting",
            "adetailer",
            "segment_anything"
        ]

        self.config_list = {}

        self.config_list["main"] = [
            ConfigType.DefaultCheckPoint,
            ConfigType.LoraEnv,
        ]
        self.config_list["lora_generate_tag"] = [
            ConfigType.LoraGenerateTag_EnableCharacter,
            ConfigType.LoraGenerateTag_EnableStyle,
            ConfigType.LoraGenerateTag_EnablePose,
            ConfigType.LoraGenerateTag_EnableItem,
            ConfigType.LoraGenerateTag_TagThCharacter,
            ConfigType.LoraGenerateTag_TagThStyle,
            ConfigType.LoraGenerateTag_TagThPose,
            ConfigType.LoraGenerateTag_TagThItem,
            ConfigType.LoraGenerateTag_ProhibitedTagsCharacter,
            ConfigType.LoraGenerateTag_ProhibitedTagsStyle,
            ConfigType.LoraGenerateTag_ProhibitedTagsPose,
            ConfigType.LoraGenerateTag_ProhibitedTagsItem,
        ]
        self.config_list["lora_block_weight"] = [
            ConfigType.LoraBlockWeight_EnableLbwCharacter,
            ConfigType.LoraBlockWeight_PresetCharacter,
            ConfigType.LoraBlockWeight_SssCharacter,
            ConfigType.LoraBlockWeight_SssValueCharacter,
            ConfigType.LoraBlockWeight_EnableLbwCharacter2,
            ConfigType.LoraBlockWeight_PresetCharacter2,
            ConfigType.LoraBlockWeight_SssCharacter2,
            ConfigType.LoraBlockWeight_SssValueCharacter2,

            ConfigType.LoraBlockWeight_EnableLbwStyle,
            ConfigType.LoraBlockWeight_PresetStyle,
            ConfigType.LoraBlockWeight_SssStyle,
            ConfigType.LoraBlockWeight_SssValueStyle,
            ConfigType.LoraBlockWeight_EnableLbwStyle2,
            ConfigType.LoraBlockWeight_PresetStyle2,
            ConfigType.LoraBlockWeight_SssStyle2,
            ConfigType.LoraBlockWeight_SssValueStyle2,

            ConfigType.LoraBlockWeight_EnableLbwPose,
            ConfigType.LoraBlockWeight_PresetPose,
            ConfigType.LoraBlockWeight_SssPose,
            ConfigType.LoraBlockWeight_SssValuePose,
            ConfigType.LoraBlockWeight_EnableLbwPose2,
            ConfigType.LoraBlockWeight_PresetPose2,
            ConfigType.LoraBlockWeight_SssPose2,
            ConfigType.LoraBlockWeight_SssValuePose2,

            ConfigType.LoraBlockWeight_EnableLbwItem,
            ConfigType.LoraBlockWeight_PresetItem,
            ConfigType.LoraBlockWeight_SssItem,
            ConfigType.LoraBlockWeight_SssValueItem,
            ConfigType.LoraBlockWeight_EnableLbwItem2,
            ConfigType.LoraBlockWeight_PresetItem2,
            ConfigType.LoraBlockWeight_SssItem2,
            ConfigType.LoraBlockWeight_SssValueItem2,
        ]
        self.config_list["generation_setting_txt2img"] = [
            ConfigType.GenTxt2Img_EnableHr,
            ConfigType.GenTxt2Img_DenoisingStr,
            ConfigType.GenTxt2Img_1pWidth,
            ConfigType.GenTxt2Img_1pHeight,
            ConfigType.GenTxt2Img_HrScale,
            ConfigType.GenTxt2Img_HrUpscaler,
            ConfigType.GenTxt2Img_Hr2pSteps,
            ConfigType.GenTxt2Img_HrResizeX,
            ConfigType.GenTxt2Img_HrResizeY,
        ]
        self.config_list["generation_setting_img2img"] = [
            ConfigType.GenImg2Img_ResizeMode,
            ConfigType.GenImg2Img_DenoisingStr,
            ConfigType.GenImg2Img_ImgCfgScale,
            ConfigType.GenImg2Img_MaskBlur,
            ConfigType.GenImg2Img_InpaintingFill,
            ConfigType.GenImg2Img_InpaintFullRes,
            ConfigType.GenImg2Img_InpaintFullResPadding,
            ConfigType.GenImg2Img_InpaintingMaskInvert,
            ConfigType.GenImg2Img_InitialNoiseMul,
        ]
        self.config_list["generation_setting_common"] = [
            ConfigType.CommonGen_SamplerName,
            ConfigType.CommonGen_Scheduler,
            ConfigType.CommonGen_Steps,
            ConfigType.CommonGen_CfgScale,
            ConfigType.CommonGen_Width,
            ConfigType.CommonGen_Height,
            ConfigType.CommonGen_RestoreFaces,
            ConfigType.CommonGen_Tiling,
            ConfigType.CommonGen_DnSaveSamples,
            ConfigType.CommonGen_DnSaveGrid,
            ConfigType.CommonGen_NegPrompt,
            ConfigType.CommonGen_Eta,
            ConfigType.CommonGen_SendImages,
            ConfigType.CommonGen_SaveImages,
        ]
        self.config_list["prompt_gen_setting"] = [
            ConfigType.PromptGen_ModelName,
            ConfigType.PromptGen_Text,
            ConfigType.PromptGen_MinLength,
            ConfigType.PromptGen_MaxLength,
            ConfigType.PromptGen_NumBeams,
            ConfigType.PromptGen_Temp,
            ConfigType.PromptGen_RepPenalty,
            ConfigType.PromptGen_LengthPref,
            ConfigType.PromptGen_SamplingMode,
            ConfigType.PromptGen_TopK,
            ConfigType.PromptGen_TopP,
        ]
        self.config_list["overwrite_generation_setting"] = [
            ConfigType.OWGen_OWSteps,
            ConfigType.OWGen_OWSamplerName,
            ConfigType.OWGen_OWScheduler,
            ConfigType.OWGen_OWCfgScale,
            ConfigType.OWGen_OWWidth,
            ConfigType.OWGen_OWHeight,
            ConfigType.OWGen_OWPrompt,
            ConfigType.OWGen_OWNegPrompt,
            ConfigType.OWGen_OWSeed,
            ConfigType.OWGen_AddLora,
            ConfigType.OWGen_AddPromptGen,
        ]
        self.config_list["adetailer"] = [
            ConfigType.Adetailer_Model,
            ConfigType.Adetailer_Prompt,
            ConfigType.Adetailer_NegPrompt,
            ConfigType.Adetailer_Model2,
            ConfigType.Adetailer_Prompt2,
            ConfigType.Adetailer_NegPrompt2,
            ConfigType.Adetailer_Model3,
            ConfigType.Adetailer_Prompt3,
            ConfigType.Adetailer_NegPrompt3,
        ]
        self.config_list["segment_anything"] = [
            ConfigType.Sam_SamModelName,
            ConfigType.Sam_DinoModelName,
        ]

    def get_opts(self, t : ConfigType):
        if t == ConfigType.DefaultCheckPoint:
            opt = self.checkpoint_list
        elif t == ConfigType.LoraEnv:
            opt = self.lora_dir_env_list
        elif t == ConfigType.GenTxt2Img_HrUpscaler:
            opt = self.latent_upscale_mode_list
        elif t == ConfigType.CommonGen_SamplerName:
            opt = self.sampler_list
        elif t == ConfigType.CommonGen_Scheduler:
            opt = self.scheduler_list
        elif t == ConfigType.Sam_SamModelName:
            opt = self.sam_model_list
        else:
            opt = DD_OPTION_MAP.get(t, None)
        return opt

    def load(self):
        super().load()
        self.info["main"] = {}
        self.info["main"]["default_checkpoint"] = self.info.pop("default_checkpoint")
        self.info["main"]["lora_dir_env"] = self.info.pop("lora_dir_env")

        for word in ["character", "character2", "style", "style2", "pose", "pose2", "item", "item2"]:
            tmp = self.info["lora_block_weight"].pop(word)
            self.info["lora_block_weight"][word + "_enable_lbw"] = tmp.pop("enable_lbw")
            self.info["lora_block_weight"][word + "_preset"] = tmp.pop("preset")
            self.info["lora_block_weight"][word + "_start_stop_step"] = tmp.pop("start_stop_step")
            self.info["lora_block_weight"][word + "_start_stop_step_value"] = tmp.pop("start_stop_step_value")
    
        tmp = [
            {
                "ad_model": "",
                "ad_prompt": "",
                "ad_negative_prompt": "",
            },
            {
                "ad_model": "",
                "ad_prompt": "",
                "ad_negative_prompt": "",
            },
            {
                "ad_model": "",
                "ad_prompt": "",
                "ad_negative_prompt": "",
            },
        ]

        tmp2 = self.info.pop("adetailer")
        for i,t in enumerate(tmp2):
            tmp[i] = t
        
        self.info["adetailer"] = {}

        for t, word in zip(tmp, ["1", "2", "3"]):
            self.info["adetailer"][word + "_ad_model"] = t["ad_model"]
            self.info["adetailer"][word + "_ad_prompt"] = t["ad_prompt"]
            self.info["adetailer"][word + "_ad_negative_prompt"] = t["ad_negative_prompt"]


    def save(self):
        mod_info = deepcopy(self.info)
        m = mod_info.pop("main")
        for key in m:
            mod_info[key] = m[key]

        ######
        m = mod_info.pop("lora_block_weight")
        mod_info["lora_block_weight"] = {}

        for word in ["character", "character2", "style", "style2", "pose", "pose2", "item", "item2"]:
            mod_info["lora_block_weight"][word] = {}

        for key in m:
            w = key.split("_")
            w1 = w[0]
            w2 = "_".join(w[1:])
            mod_info["lora_block_weight"][w1][w2] = m[key]
        
        ######
        m = mod_info.pop("adetailer")
        mod_info["adetailer"] = []

        valid = []
        for word in ["1_ad_model", "2_ad_model", "3_ad_model"]:
            if m[word] not in ("none", ""):
                w = word.split("_")
                w1 = w[0]
                valid.append(w1)

        for num in valid:
            mod_info["adetailer"].append(
                {
                    "ad_model" : m[num + "_ad_model"],
                    "ad_prompt" : m[num + "_ad_prompt"],
                    "ad_negative_prompt" : m[num + "_ad_negative_prompt"],
                }
            )


        json_text = json.dumps(mod_info, indent=4, ensure_ascii=False)
        self.json_path.write_text(json_text, encoding="utf-8")


class GenerateEditViewCreator(TabViewCreator):
    edit_view_input = {}

    def __init__(self, page):

        Path("input/gui_tmp").mkdir(parents=True, exist_ok=True)
        tmp_json_path = Path("input/gui_tmp") / Path( get_time_str() + "_input" +".json" )
        tmp_json_path = str(tmp_json_path)

        super().__init__(page, "/view_generate_edit", "Generate", tmp_json_path, "/view_generate")

        self.controlnet_type_list = get_controlnet_type_list()
        self.checkpoint_list = get_checkpoint_list()
        self.sampler_list = get_sampler_list()
        self.scheduler_list = get_scheduler_list()

        self.tab_order = [
            "common",
            "seq",
        ]

        self.config_list = {}

        self.config_list["common"] = [
            ConfigType.Input_CheckPoint,
            ConfigType.Input_Seed,
        ]
        self.config_list["prompt"] = [
            ConfigType.Input_Prompt_CharacterLora,
            ConfigType.Input_Prompt_CharacterLora2,
            ConfigType.Input_Prompt_StyleLora,
            ConfigType.Input_Prompt_StyleLora2,
            ConfigType.Input_Prompt_PoseLora,
            ConfigType.Input_Prompt_PoseLora2,
            ConfigType.Input_Prompt_ItemLora,
            ConfigType.Input_Prompt_ItemLora2,
            ConfigType.Input_Prompt_PresetTags,
            ConfigType.Input_Prompt_Header,
            ConfigType.Input_Prompt_Footer,
        ]

        self.config_list["generation_setting"] = [
            ConfigType.CommonGen_SamplerName,
            ConfigType.CommonGen_Scheduler,
            ConfigType.CommonGen_Steps,
            ConfigType.CommonGen_CfgScale,
            ConfigType.CommonGen_Width,
            ConfigType.CommonGen_Height,
            ConfigType.CommonGen_NegPrompt,
            ConfigType.GenImg2Img_DenoisingStr,
        ]

        self.config_list["prompt_gen"] = [
            ConfigType.Input_PromptGen_Type,
            ConfigType.PromptGen_Text,
            ConfigType.PromptGen_MaxLength,
            ConfigType.Input_PromptGen_IsFooter,
        ]

        self.config_list["overwrite_generation_setting"] = [
            ConfigType.Input_OW_PngInfo,
            ConfigType.OWGen_AddLora,
            ConfigType.OWGen_AddPromptGen,
        ]

        self.config_list["seq"] = [
            ConfigType.Input_Seq_Type,
            ConfigType.Input_Seq_InputImage,
            ConfigType.Input_Seq_OutputScale,
        ]

        self.config_list["controlnet"] = [
            ConfigType.Input_Seq_ControlnetType,
            ConfigType.Input_Seq_ControlnetImage,
            ConfigType.Input_Seq_ControlnetCNTarget,
            ConfigType.Controlnet_Weight,
            ConfigType.Controlnet_GuidanceStart,
            ConfigType.Controlnet_GuidanceEnd,
        ]

        self.config_list["seq_generation_setting"] = [
            ConfigType.GenImg2Img_DenoisingStr,
        ]

        self.updated = False


    def is_update(self):
        def is_update_inner(c):
            if type(c) == dict:
                for name in c:
                    if is_update_inner(c[name]):
                        return True
                return False
            else:
                return c.is_update()
        
        if self.updated:
            return True
        
        result = is_update_inner(self.config_items)

        return result

    def get_opts(self, t : ConfigType):
        if t == ConfigType.Input_CheckPoint:
            opt = self.checkpoint_list
        elif t == ConfigType.CommonGen_SamplerName:
            opt = self.sampler_list
        elif t == ConfigType.CommonGen_Scheduler:
            opt = self.scheduler_list
        elif t == ConfigType.Input_Seq_ControlnetType:
            opt = self.controlnet_type_list
        else:
            opt = DD_OPTION_MAP.get(t, None)
        return opt


    def load(self):
        self.info = deepcopy(GenerateEditViewCreator.edit_view_input)
    

    def load_config_item_common_main(self):
        from sd_batch_runner.util import config_get_default_checkpoint
        default_value = {
            "checkpoint" : config_get_default_checkpoint(),
            "seed" : "@random",
        }
    
        cur_info = self.info.get("common", {})

        self.config_items["common"]["main"] = {}

        for t in self.config_list["common"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = cur_info.get(name, v)
            self.config_items["common"]["main"][name] = ConfigItem.create( t, v, opt, self.page)

    def load_config_item_common_prompt(self):
        default_value = {
            "character_lora" : ["", 1.0],
            "character_lora2" : ["", 1.0],
            "style_lora" : ["", 1.0],
            "style_lora2" : ["", 1.0],
            "pose_lora" : ["", 1.0],
            "pose_lora2" : ["", 1.0],
            "item_lora" : ["", 1.0],
            "item_lora2" : ["", 1.0],
            "preset_tags" : [],
            "header" : "",
            "footer" : "",
        }
    
        cur_info = self.info.get("common", {}).get("prompt",{})

        self.config_items["common"]["prompt"] = {}

        for t in self.config_list["prompt"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = cur_info.get(name, v)
            self.config_items["common"]["prompt"][name] = ConfigItem.create( t, v, opt, self.page)

    def load_config_item_common_gen(self):
        from sd_batch_runner.util import config_get_default_generation_setting
        def_conf_gen = config_get_default_generation_setting(False)
        default_value = {
            "sampler_name" : def_conf_gen["sampler_name"],
            "scheduler" : def_conf_gen["scheduler"],
            "steps" : def_conf_gen["steps"],
            "cfg_scale" : def_conf_gen["cfg_scale"],
            "width" : def_conf_gen["width"],
            "height" : def_conf_gen["height"],
            "negative_prompt" : def_conf_gen["negative_prompt"],
            "denoising_strength" : def_conf_gen["denoising_strength"],
        }
    
        cur_info = self.info.get("common", {}).get("generation_setting",{})

        self.config_items["common"]["generation_setting"] = {}

        for t in self.config_list["generation_setting"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = cur_info.get(name, v)
            self.config_items["common"]["generation_setting"][name] = ConfigItem.create( t, v, opt, self.page)


    def load_config_item_common_prompt_gen(self, force = False):
        from sd_batch_runner.util import config_get_default_prompt_gen_setting
        def_conf_gen = config_get_default_prompt_gen_setting()
        default_value = {
            "type" : "any_time",
            "text" : def_conf_gen["text"],
            "max_length" : def_conf_gen["max_length"],
            "is_footer" : True,
        }
    
        cur_info = self.info.get("common", {}).get("prompt_gen",{})

        if not force:
            if not cur_info:
                return

        self.config_items["common"]["prompt_gen"] = {}

        for t in self.config_list["prompt_gen"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = cur_info.get(name, v)
            self.config_items["common"]["prompt_gen"][name] = ConfigItem.create( t, v, opt, self.page)

    def load_config_item_common_ow_gen(self, force = False):
        from sd_batch_runner.util import config_get_default_overwrite_generation_setting
        def_conf_gen = config_get_default_overwrite_generation_setting()
        default_value = {
            "png_info" : "",
            "add_lora" : def_conf_gen["add_lora"],
            "add_prompt_gen" : def_conf_gen["add_prompt_gen"],
        }
    
        cur_info = self.info.get("common", {}).get("overwrite_generation_setting",{})

        if not force:
            if not cur_info:
                return

        self.config_items["common"]["overwrite_generation_setting"] = {}

        for t in self.config_list["overwrite_generation_setting"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = cur_info.get(name, v)
            self.config_items["common"]["overwrite_generation_setting"][name] = ConfigItem.create( t, v, opt, self.page)


    def load_config_item_seq_main(self, index, c=None):
        default_value = {
            "type" : "txt2img",
            "input_image" : "",
            "output_scale" : 1.0,
        }
    
        if c:
            seq = c
        else:
            seq_list = self.info.get("seq", [])
            if len(seq_list) > index:
                seq = seq_list[index]
            else:
                seq = {"type":"txt2img"}

        self.config_items["seq"][index]["main"] = {}

        for t in self.config_list["seq"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = seq.get(name, v)
            self.config_items["seq"][index]["main"][name] = ConfigItem.create( t, v, opt, self.page)

    def add_config_item_seq_controlnet_item(self, index, item_index, c):
        from sd_batch_runner.util import get_controlnet_setting
        cur_type = c.get("type", self.controlnet_type_list[0])
        def_conf = get_controlnet_setting(cur_type)

        default_value = {
            "type" : cur_type,
            "image" : "",
            "cn_target" : "",
            "weight" : def_conf["weight"],
            "guidance_start" : def_conf["guidance_start"],
            "guidance_end" : def_conf["guidance_end"],
        }

        self.config_items["seq"][index]["controlnet"][item_index]={}

        for t in self.config_list["controlnet"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = c.get(name, v)
            self.config_items["seq"][index]["controlnet"][item_index][name] = ConfigItem.create( t, v, opt, self.page)


    def load_config_item_seq_controlnet(self, index, force=False, c= None):
    
        if c:
            controlnet_list = c
        else:
            seq_list = self.info.get("seq", [])
            if len(seq_list) > index:
                seq = seq_list[index]
            else:
                seq = {"type":"txt2img"}
        
            controlnet_list = seq.get("controlnet", [])

        if not force:
            if not controlnet_list:
                return
        
        self.config_items["seq"][index]["controlnet"] = {}
        
        for i,cont_info in enumerate(controlnet_list):
            self.add_config_item_seq_controlnet_item(index, i, cont_info)

    def load_config_item_seq_gen(self, index, force=False, c= None):
        from sd_batch_runner.util import config_get_default_generation_setting
        def_conf_gen = config_get_default_generation_setting(False)

        default_value = {
            "denoising_strength" : def_conf_gen["denoising_strength"],
        }
    
        if c:
            gen = c
        else:
            seq_list = self.info.get("seq", [])
            if len(seq_list) > index:
                seq = seq_list[index]
            else:
                seq = {"type":"txt2img"}
            gen = seq.get("generation_setting",{})

        if not force:
            if not gen:
                return

        self.config_items["seq"][index]["generation_setting"] = {}

        for t in self.config_list["seq_generation_setting"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = gen.get(name, v)
            self.config_items["seq"][index]["generation_setting"][name] = ConfigItem.create( t, v, opt, self.page)

    def load_config_item_seq_prompt(self, index, force=False, c= None):

        default_value = {
            "character_lora" : ["", 1.0],
            "character_lora2" : ["", 1.0],
            "style_lora" : ["", 1.0],
            "style_lora2" : ["", 1.0],
            "pose_lora" : ["", 1.0],
            "pose_lora2" : ["", 1.0],
            "item_lora" : ["", 1.0],
            "item_lora2" : ["", 1.0],
            "preset_tags" : [],
            "header" : "",
            "footer" : "",
        }
    
        if c:
            gen = c
        else:
            seq_list = self.info.get("seq", [])
            if len(seq_list) > index:
                seq = seq_list[index]
            else:
                seq = {"type":"txt2img"}
            gen = seq.get("prompt",{})

        if not force:
            if not gen:
                return

        self.config_items["seq"][index]["prompt"] = {}

        for t in self.config_list["prompt"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = gen.get(name, v)
            self.config_items["seq"][index]["prompt"][name] = ConfigItem.create( t, v, opt, self.page)

    def load_config_item_seq_prompt_gen(self, index, force=False, c= None):
        from sd_batch_runner.util import config_get_default_prompt_gen_setting
        def_conf_gen = config_get_default_prompt_gen_setting()
        default_value = {
            "type" : "any_time",
            "text" : def_conf_gen["text"],
            "max_length" : def_conf_gen["max_length"],
            "is_footer" : True,
        }

        if c:
            gen = c
        else:
            seq_list = self.info.get("seq", [])
            if len(seq_list) > index:
                seq = seq_list[index]
            else:
                seq = {"type":"txt2img"}
            gen = seq.get("prompt_gen",{})

        if not force:
            if not gen:
                return

        self.config_items["seq"][index]["prompt_gen"] = {}

        for t in self.config_list["prompt_gen"]:
            name = CONFIG_MAP[t][2]
            opt = self.get_opts(t)
            v = default_value[name]
            v = gen.get(name, v)
            self.config_items["seq"][index]["prompt_gen"][name] = ConfigItem.create( t, v, opt, self.page)


    def load_config_item_seq(self):

        seq_list = self.info.get("seq", [])
        seq_len = len(seq_list)
        if seq_len == 0:
            seq_len = 1
        
        for i in range(seq_len):
            self.config_items["seq"][i] = {}
            self.load_config_item_seq_main(i)
            self.load_config_item_seq_controlnet(i)
            self.load_config_item_seq_gen(i)
            self.load_config_item_seq_prompt(i)
            self.load_config_item_seq_prompt_gen(i)

    def add_config_item_seq(self):
        keys = self.config_items["seq"].keys()
        keys = list(keys)
        keys.sort()
        new_key = 0 if len(keys)==0 else keys[-1] + 1

        self.config_items["seq"][new_key] = {}
        self.load_config_item_seq_main(new_key, {"type":"txt2img"})
        return new_key

    def add_config_item_seq_controlnet(self, seq_key):
        keys = self.config_items["seq"][seq_key]["controlnet"].keys()
        keys = list(keys)
        keys.sort()
        new_key = 0 if len(keys)==0 else keys[-1] + 1

        c = {
            "type" : self.controlnet_type_list[0]
        }

        self.add_config_item_seq_controlnet_item(seq_key, new_key, c)
        return new_key

    def add_config_item_seq_option(self, seq_key, option):
        tmp = self.config_items["seq"][seq_key].get(option, {})
        if tmp:
            return None
        
        self.config_items["seq"][seq_key][option] = {}

        if option == "controlnet":
            c = [
                {
                    "type" : self.controlnet_type_list[0]
                }
            ]
            self.load_config_item_seq_controlnet(seq_key, True, c)
        elif option == "generation_setting":
            c = {}
            self.load_config_item_seq_gen(seq_key, True, c)
        elif option == "prompt":
            c = {}
            self.load_config_item_seq_prompt(seq_key, True, c)
        elif option == "prompt_gen":
            c = {}
            self.load_config_item_seq_prompt_gen(seq_key, True, c)

        return option
    
    def add_config_item_common_option(self, option):
        tmp = self.config_items["common"].get(option, {})
        if tmp:
            return None
        
        self.config_items["common"][option] = {}

        if option == "prompt_gen":
            self.load_config_item_common_prompt_gen(True)
        elif option == "overwrite_generation_setting":
            self.load_config_item_common_ow_gen(True)
        return option


    def load_config(self):

        self.config_items["common"] = {}

        self.load_config_item_common_main()
        self.load_config_item_common_prompt()
        self.load_config_item_common_gen()
        self.load_config_item_common_prompt_gen()
        self.load_config_item_common_ow_gen()

        self.config_items["seq"] = {}

        self.load_config_item_seq()
    
    def save_config(self):

        self.validate_value()

        def convert_to_info(c):
            return { key:c[key].val for key in c }

        new_info = {}
        # common
            # main
        new_info["common"] = convert_to_info(self.config_items["common"]["main"])
            # prompt
            # gen
            # prompt_gen
            # ow_gen
        for sub in ["prompt","generation_setting","prompt_gen","overwrite_generation_setting"]:
            conf = self.config_items["common"].get(sub, None)
            if conf:
                new_info["common"][sub] = convert_to_info(conf)
        
        # seq
        new_info["seq"] = []
        for key in self.config_items["seq"]:
            cur = self.config_items["seq"][key]
            # 0
                # main
            tmp = convert_to_info(cur["main"])
                # controlnet
            controlnet_dict = cur.get("controlnet", None)
            if controlnet_dict:
                c_tmp = []
                for c_key in controlnet_dict:
                    c_cur = controlnet_dict[c_key]
                    c_tmp.append(convert_to_info(c_cur))
                tmp["controlnet"] = c_tmp
                # gen
            for sub in ["prompt","generation_setting","prompt_gen"]:
                conf = cur.get(sub, None)
                if conf:
                    tmp[sub] = convert_to_info(conf)
            new_info["seq"].append(tmp)

        self.info = new_info

        self.updated = False


    def create_config_item(self, c, key=None):
        raise NotImplementedError()

    def create_inner_panel(self, c, key_list):

        exp = ft.ExpansionPanel(
            bgcolor=ft.colors.GREEN_50,
            can_tap_header=True,
            header=ft.ListTile(title=ft.Text(key_list[-1])),
            data=(key_list)
        )

        if len(key_list) == 3:
            if key_list[-3] == "seq" and key_list[-1] == "main":
                exp.expanded = True

        controls = [ c[name].create_ui_control() for name in c ]

        if key_list[-2] == "controlnet":
            footer = self.create_tab_tail_del(exp,key_list)
            controls.append(footer)
        elif key_list[-1] in ("prompt_gen","overwrite_generation_setting"):
            footer = self.create_tab_tail_del(exp,key_list)
            controls.append(footer)
        elif key_list[0] == "seq" and key_list[-1] in("prompt", "prompt_gen", "generation_setting"):
            footer = self.create_tab_tail_del(exp,key_list)
            controls.append(footer)


        exp.content = ft.Container(
            ft.Column(
                controls=controls,
                horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                tight=True
            ),
            margin=0,
            padding=10,
            border_radius=0,
        )

        return exp
    
    def create_nested_panel(self, c, key_list):
        exp = ft.ExpansionPanel(
            bgcolor=ft.colors.GREEN_100,
            can_tap_header=True,
            header=ft.ListTile(title=ft.Text(key_list[-1])),
            data=(key_list)
        )

        nested_panel_list = ft.ExpansionPanelList(
            expand_icon_color=ft.colors.BLACK,
            elevation=8,
            divider_color=ft.colors.BLACK,
            controls=[
            ],
            data=(key_list)
        )

        #logger.info(f"{c=} {key=}")

        for i_key in c:
            if i_key == "controlnet":
                panel = self.create_nested_panel(c[i_key], key_list + [i_key])
            else:
                panel = self.create_inner_panel(c[i_key], key_list + [i_key])
            nested_panel_list.controls.append(panel)



        if key_list[-1] == "controlnet":
            footer1 = self.create_tab_tail_add(nested_panel_list, key_list)
            footer2 = self.create_tab_tail_del(exp, key_list)
            exp.content = ft.Container(
                ft.Column(
                    controls=[nested_panel_list, footer1, footer2],
                    horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                    tight=True
                ),
                margin=5,
                padding=5,
            )
        elif key_list[-2] == "seq":
            footer1 = self.create_tab_tail_add_option(nested_panel_list, key_list)
            footer2 = self.create_tab_tail_del(exp, key_list)
            exp.content = ft.Container(
                ft.Column(
                    controls=[nested_panel_list, footer1, footer2],
                    horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                    tight=True
                ),
                margin=5,
                padding=5,
            )
        else:
            exp.content = ft.Container(
                nested_panel_list,
                margin=5,
                padding=5,
            )

        return exp
    
    def create_inner_tab(self, c, key_list):

        tab = ft.Tab(
            text=key_list[-1],
        )

        nested_panel_list = ft.ExpansionPanelList(
            expand_icon_color=ft.colors.BLACK,
            elevation=8,
            divider_color=ft.colors.BLACK,
            controls=[
            ],
            data=(key_list)
        )

        #logger.info(f"{c=} {key=}")

        for i_key in c:
            if i_key == "controlnet":
                panel = self.create_nested_panel(c[i_key], key_list + [i_key])
            else:
                panel = self.create_inner_panel(c[i_key], key_list + [i_key])
            nested_panel_list.controls.append(panel)


        if key_list[-2] == "seq":
            footer1 = self.create_tab_tail_add_option(nested_panel_list, key_list)
            footer2 = self.create_tab_tail_del(tab, key_list)
            tab.content = ft.Container(
                ft.Column(
                    controls=[nested_panel_list, footer1, footer2],
                    horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
                    tight=True
                ),
                margin=5,
                padding=5,
            )
        else:
            tab.content = ft.Container(
                nested_panel_list,
                margin=5,
                padding=5,
            )

        return tab

    
    def create_tab_tail_add(self, panel_or_tab_list, key_list):
        def append_item_seq_tab(e: ft.ControlEvent):
            new_key = self.add_config_item_seq()
            panel = self.create_inner_tab(self.config_items["seq"][new_key], ["seq",new_key])
            tail = panel_or_tab_list.tabs.pop()
            panel_or_tab_list.tabs.append(panel)
            panel_or_tab_list.tabs.append(tail)
            self.page.update()
            self.updated = True

        def append_item_controlnet(e: ft.ControlEvent):
            new_key = self.add_config_item_seq_controlnet(key_list[-2])
            panel = self.create_inner_panel(self.config_items["seq"][key_list[-2]]["controlnet"][new_key], key_list + [new_key])
            panel_or_tab_list.controls.append(panel)
            self.page.update()
            self.updated = True

        if key_list[-1] == "seq":
            append_item = append_item_seq_tab

            return ft.Tab( tab_content=
                ft.IconButton(icon=ft.icons.ADD, on_click=append_item),
            )

        elif key_list[-1] == "controlnet":
            append_item = append_item_controlnet

            return ft.Row( controls=[
                ft.IconButton(icon=ft.icons.ADD, width=200, on_click=append_item),
            ] )
    
    def create_tab_tail_add_option(self, panel_list:ft.ExpansionPanelList, key_list):
        def append_option_seq(e: ft.ControlEvent):
            new_key = self.add_config_item_seq_option(key_list[-1], dd.value )
            if not new_key:
                return
            
            if new_key == "controlnet":
                panel = self.create_nested_panel(self.config_items["seq"][key_list[-1]][new_key], key_list + [new_key])
            else:
                panel = self.create_inner_panel(self.config_items["seq"][key_list[-1]][new_key], key_list + [new_key])

            panel_list.controls.append(panel)
            self.page.update()
            self.updated = True

        def append_option_common(e: ft.ControlEvent):
            new_key = self.add_config_item_common_option(dd.value )
            if not new_key:
                return

            panel = self.create_inner_panel(self.config_items["common"][new_key], key_list + [new_key])
            panel_list.controls.append(panel)
            self.page.update()
            self.updated = True



        if key_list[0] == "seq":
            append_item = append_option_seq
            opt_label = ["controlnet","prompt","generation_setting","prompt_gen"]
        elif key_list[0] == "common":
            append_item = append_option_common
            opt_label = ["prompt_gen","overwrite_generation_setting"]


        opt = [ft.dropdown.Option(o) for o in opt_label]
        dd = ft.Dropdown(opt_label[0], width=300, options=opt)

        return ft.Row( controls=[
            ft.IconButton(icon=ft.icons.ADD, width=100, on_click=append_item),
            dd
        ] )


    def create_tab_tail_del(self, panel_or_tab, key_list):

        def delete_item_seq(e: ft.ControlEvent):
            if len(self.config_items["seq"]) == 1:
                return
            panel_or_tab.parent.tabs.remove(panel_or_tab)
            self.config_items["seq"].pop(key_list[-1])
            self.page.update()
            self.updated = True

        def delete_item_controlnet_item(e: ft.ControlEvent):
            logger.info(f"{key_list=}")
            if len(self.config_items["seq"][key_list[-3]]["controlnet"]) == 1:
                return
            panel_or_tab.parent.controls.remove(panel_or_tab)
            self.config_items["seq"][key_list[-3]]["controlnet"].pop(key_list[-1])
            self.page.update()
            self.updated = True

        def delete_item_common(e: ft.ControlEvent):
            panel_or_tab.parent.controls.remove(panel_or_tab)
            last_key = key_list.pop(-1)
            cur = self.config_items
            for key in key_list:
                cur = cur[key]
            cur.pop(last_key)
            self.page.update()
            self.updated = True

        if key_list[-2] == "seq":
            delete_item = delete_item_seq
        elif key_list[-2] == "controlnet":
            delete_item = delete_item_controlnet_item
        else:
            delete_item = delete_item_common

        return ft.Row( controls=[
            ft.IconButton(icon=ft.icons.DELETE, width=200, on_click=delete_item),
        ] )

    def show_tab(self, key):

        c = self.config_items[key]

        tab = ft.Tab(
            text=key
        )


        if key == "common":
            panel_list = ft.ExpansionPanelList(
                expand_icon_color=ft.colors.BLACK,
                elevation=8,
                divider_color=ft.colors.BLACK,
                controls=[
                ],
                data=(key)
            )

            for i_key in c:
                panel = self.create_inner_panel(c[i_key], [key, i_key])
                panel_list.controls.append(panel)
            
            controls=[panel_list]

            footer = self.create_tab_tail_add_option(panel_list, [key])
            controls=[panel_list, footer]

        else:

            tab_list = ft.Tabs(
                selected_index=0,
                animation_duration=300,
                expand=True,
                divider_color=ft.colors.BLACK,
                indicator_color=ft.colors.BLACK,
                data=(key)
            )

            for i_key in c:
                i_tab = self.create_inner_tab(c[i_key], [key, i_key])
                tab_list.tabs.append(i_tab)
            
            footer = self.create_tab_tail_add(tab_list,[key])
            tab_list.tabs.append( footer)

            controls=[tab_list]

        tab.content=ft.Column(
            controls=controls,
            horizontal_alignment= ft.CrossAxisAlignment.STRETCH,
            scroll=ft.ScrollMode.ALWAYS,
            expand=True,
            adaptive=True,
            tight=True
        )


        self.tab_list.tabs.append(tab)

    def create_tail_icon(self):
        def warn_save(e: ft.ControlEvent):
            self.warn_save()
        
        batch_counter = ft.TextField("1", label="batch_count", width=100, input_filter=ft.InputFilter(regex_string=r"^(\d*)$", allow=True))

        def on_generate_start(e: ft.ControlEvent):
            self.save_config()
            self.save()
            self.notify_updated()

            v = int(batch_counter.value)
            self.page.data["generate_batch_count"] = v if v > 0 else 1
            self.page.data["input_json_path"] = str(self.json_path)
            self.page.go("/view_generate_progress")

        return ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.FilledButton(text="Generate", width=200, on_click=on_generate_start),
                            batch_counter,
                            ft.VerticalDivider(width=100, thickness=30),
                            ft.IconButton(icon=ft.icons.SAVE, width=50, on_click=warn_save),
                            ft.Text( self.json_path.absolute() )
                        ]
                    ),
                )


def create_generate_view(page: ft.Page):

    def create_new(e: ft.ControlEvent):
        
        from sd_batch_runner.util import config_clear_cache,config_get_default_generation_setting
        config_clear_cache()
        lora_clear_cache()

        gen = config_get_default_generation_setting(True)

        GenerateEditViewCreator.edit_view_input = {
            "common" : {
                "prompt":{
                    "character_lora" : ["@random_per_seq",1.0],
                    "character_lora2" : "",
                    "style_lora" : "",
                    "style_lora2" : "",
                    "pose_lora" : "",
                    "pose_lora2" : "",
                    "item_lora" : "",
                    "item_lora2" : "",
                    "header" : "",
                    "footer" : "",
                },
                "seed" : "@random",
                "generation_setting" : {
                    "width" : gen["width"],
                    "height" : gen["height"],
                    "negative_prompt" : gen["negative_prompt"],
                },
            },
            "seq" : [
                {
                    "type" : "txt2img"
                }
            ]
        }
        page.go("/view_generate_edit")
    def load(e: ft.ControlEvent):
        from sd_batch_runner.util import config_clear_cache
        config_clear_cache()
        lora_clear_cache()

        def on_pick(e: ft.FilePickerResultEvent):
            if e.files:
                json_path = Path(e.files[0].path)

                logger.info(f"{json_path=}")

                if json_path.is_file():
                    with open(json_path, "r", encoding="utf-8") as f:
                        GenerateEditViewCreator.edit_view_input = json.load(f)
                        page.go("/view_generate_edit")
                else:
                    logger.info(f"{json_path} not found")
            
            page.overlay.remove(pick_files_dialog)
            page.update()

        pick_files_dialog = ft.FilePicker(on_result=on_pick)
        page.overlay.append(pick_files_dialog)
        page.update()
        pick_files_dialog.pick_files(
            dialog_title="Select input json file",
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions = ["json","JSON"]
        )


    update_lora_is_overwrite = ft.Checkbox(value=False, label="is_overwrite",width=150)

    def on_click(e: ft.ControlEvent):

        cancel_dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Text("Lora Update"),
            content=ft.Text("Updating..."),
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: page.add(
                ft.Text("Modal dialog dismissed"),
            ),
        )

        page.open(cancel_dlg_modal)
        update_lora_command(update_lora_is_overwrite.value)
        page.close(cancel_dlg_modal)
        

    view = ft.View("/view_generate", [
        ft.AppBar(
            leading=ft.IconButton(icon=ft.icons.ARROW_BACK, width=200, on_click=lambda _: page.go("/view_top")),
            title=ft.Text("Generate"),
            bgcolor=ft.colors.BLUE),
        ft.FilledButton(text = "Create New Sequence", width=300, on_click=create_new),
        ft.FilledButton(text = "Load Sequence", width=300, on_click=load),
        ft.Row([
            ft.OutlinedButton(text = "Lora Update", width=150, on_click=on_click),
            update_lora_is_overwrite
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            alignment = ft.MainAxisAlignment.CENTER
            ),
        ],
        vertical_alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment = ft.CrossAxisAlignment.CENTER
    )
    
    return view

def create_generate_progress_view(page: ft.Page):
    import threading

    lock = threading.Lock()

    def warn_cancel(e: ft.ControlEvent):
        with lock:
            if completed:
                lora_clear_cache()
                page.go("/view_generate_edit")
                return

            def handle_close(e):
                with lock:
                    page.close(cancel_dlg_modal)

            def do_cancel(e: ft.ControlEvent):
                with lock:
                    status_txt.value = "Canceling..."
                    cancel_generate()
                    page.close(cancel_dlg_modal)
        
            cancel_dlg_modal = ft.AlertDialog(
                modal=True,
                title=ft.Text("Please confirm"),
                content=ft.Text("Do you really want to Cancel Generation?"),
                actions=[
                    ft.TextButton("Yes", on_click=do_cancel),
                    ft.TextButton("No", on_click=handle_close)
                ],
                actions_alignment=ft.MainAxisAlignment.END,
                on_dismiss=lambda e: page.add(
                    ft.Text("Modal dialog dismissed"),
                ),
            )
            page.open(cancel_dlg_modal)


    def open_dir(e: ft.ControlEvent):
        with lock:
            page.launch_url(output_dir_txt.value)
    

    big_label_style= ft.TextStyle( size=30 )


    completed = False
    status_txt = ft.Text("Generating...", style=big_label_style)
    progress_txt = ft.Text(f"{0}%", style=big_label_style)
    progress = ft.ProgressBar(width = 500, bar_height=50, value=0)
    output_dir_txt = ft.Text("")

    view = ft.View("/view_generate_progress", [
            ft.AppBar(
                leading=ft.IconButton(icon=ft.icons.ARROW_BACK, width=200, on_click=warn_cancel),
                title=ft.Text("Generating..."),
                bgcolor=ft.colors.BLUE),
            ft.Row([status_txt,progress_txt], alignment=ft.MainAxisAlignment.CENTER),
            progress,
            ft.Divider(height=100, thickness=0, opacity=0.0),
            ft.FilledButton(text = "Open Output Directory", width=200, on_click=open_dir),
            output_dir_txt
        ],
        vertical_alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment = ft.CrossAxisAlignment.CENTER
    )


    # page.data["generate_batch_count"]
    # page.data["input_json_path"]

    def on_progress(v):
        with lock:
            progress.value = v
            progress_txt.value = f"{int(v*100)}%"
            page.update()

    def on_complete(msg):
        nonlocal completed
        with lock:
            if msg == "Success":
                progress.value = 1.0
            status_txt.value = msg
            completed = True
            page.update()
    

    outputdir = async_generate(
        Path(page.data["input_json_path"]),
        page.data["generate_batch_count"],
        on_progress,
        on_complete)
    
    output_dir_txt.value = str(outputdir)

    
    return view


_edit_view_cache = None

def main(page: ft.Page):
    from sd_batch_runner.util import config_restore_files_if_needed
    config_restore_files_if_needed()

    page.title = "SD Batch Runner"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.data = {}

    def create_top_view(page: ft.Page):
        return ft.View("/view_top", [
            ft.FilledButton(text = "Generate", width=300, on_click=lambda _: page.go("/view_generate")),
            ft.OutlinedButton(text = "Lora Directory Setting", width=300, on_click=lambda _: page.go("/view_lora")),
            ft.OutlinedButton(text = "Preset Tags Setting", width=300, on_click=lambda _: page.go("/view_preset_tags")),
            ft.OutlinedButton(text = "Controlnet Alias Setting", width=300, on_click=lambda _: page.go("/view_controlnet")),
            ft.OutlinedButton(text = "Config", width=300, on_click=lambda _: page.go("/view_config")),],
            vertical_alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment = ft.CrossAxisAlignment.CENTER
        )

    def route_change(handler):
        # /view_top
        #   -> /view_generate
        #       -> /view_generate_edit
        #           -> / view_generate_progress
        #   -> /view_lora
        #   -> /view_preset_tags
        #   -> /view_controlnet
        #   -> /view_view_config
        #
        global _edit_view_cache
        troute = ft.TemplateRoute(handler.route)

        page.views.clear()


        if troute.match("/view_top"):
            page.views.append(create_top_view(page))
        elif troute.match("/view_generate"):
            _edit_view_cache = None
            page.views.append(create_generate_view(page))
        elif troute.match("/view_generate_edit"):
            if _edit_view_cache == None:
                _edit_view_cache = GenerateEditViewCreator(page).create()
            page.views.append(_edit_view_cache)
        elif troute.match("/view_lora"):
            page.views.append(LoraEnvViewCreator(page).create())
        elif troute.match("/view_preset_tags"):
            page.views.append(PresetTagsViewCreator(page).create())
        elif troute.match("/view_controlnet"):
            page.views.append(ControlnetViewCreator(page).create())
        elif troute.match("/view_config"):
            page.views.append(ConfigViewCreator(page).create())
        elif troute.match("/view_generate_progress"):
            page.views.append(create_generate_progress_view(page))
        page.update()

    page.on_route_change = route_change

    page.go("/view_top")


ft.app(main)