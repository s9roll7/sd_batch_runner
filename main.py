import logging
import time
from pathlib import Path
import fire

from sd_batch_runner.util import *
from sd_batch_runner.lora import update_lora_command,show_lora_command,show_lora_env_command,set_lora_env_command
from sd_batch_runner.generate import one_command,generate_command,show_checkpoint_command,set_default_checkpoint_command,show_controlnet_command

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)



###############################################################

class Command:
    def __init__(self):
        config_restore_files_if_needed()

    def update_lora(self, is_overwrite=False):
        start_tim = time.time()

        update_lora_command(is_overwrite)

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")
    
    def one(self,char="@random",style="@random",pose="@random",item=None,header="score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, masterpiece, perfect face, perfect eyes",footer="zPDXL3",n=1):
        start_tim = time.time()

        one_command(char,style,pose,item,header,footer,n)
        
        logger.info(f"Total Elapsed time : {time.time() - start_tim}")
    
    def generate(self, json_path, n=1):
        start_tim = time.time()

        generate_command(Path(json_path),n)

        clear_video_cache()

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")

    def show_checkpoint(self):
        start_tim = time.time()

        show_checkpoint_command()

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")

    def set_default_checkpoint(self, checkpoint_number):
        start_tim = time.time()

        set_default_checkpoint_command(checkpoint_number)

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")

    def show_controlnet(self):
        start_tim = time.time()

        show_controlnet_command()

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")
    
    def show_lora(self):
        start_tim = time.time()

        show_lora_command()

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")

    def show_lora_env(self):
        start_tim = time.time()

        show_lora_env_command()

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")

    def set_lora_env(self, new_env):
        start_tim = time.time()

        set_lora_env_command(new_env)

        logger.info(f"Total Elapsed time : {time.time() - start_tim}")

fire.Fire(Command)
