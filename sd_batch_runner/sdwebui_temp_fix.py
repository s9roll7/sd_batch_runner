import json
import logging
import time
from pathlib import Path
from datetime import datetime
import math
import random
import re

import webuiapi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

#########################################################
# sdwebui temp fix

class ControlNetUnit2(webuiapi.ControlNetUnit):

    def to_dict(self):
        if not hasattr(self, 'effective_region_mask'):
            self.effective_region_mask = None

        if self.image is None and self.mask is None:
            return {
                "module": self.module,
                "model": None if self.model=="none" else self.model,
                "weight": self.weight,
                "resize_mode": self.resize_mode,
                "low_vram": self.low_vram,
                "processor_res": self.processor_res,
                "threshold_a": self.threshold_a,
                "threshold_b": self.threshold_b,
                "guidance_start": self.guidance_start,
                "guidance_end": self.guidance_end,
                "control_mode": self.control_mode,
                "pixel_perfect": self.pixel_perfect,
                "hr_option": self.hr_option,
                "enabled": self.enabled,
            }
        else:
            return {
                "image": webuiapi.raw_b64_img(self.image) if self.image else "",
                "mask": webuiapi.raw_b64_img(self.mask) if self.mask is not None else None,
                "effective_region_mask": webuiapi.raw_b64_img(self.effective_region_mask) if self.effective_region_mask is not None else None,
                "module": self.module,
                "model": None if self.model=="none" else self.model,
                "weight": self.weight,
                "resize_mode": self.resize_mode,
                "low_vram": self.low_vram,
                "processor_res": self.processor_res,
                "threshold_a": self.threshold_a,
                "threshold_b": self.threshold_b,
                "guidance_start": self.guidance_start,
                "guidance_end": self.guidance_end,
                "control_mode": self.control_mode,
                "pixel_perfect": self.pixel_perfect,
                "hr_option": self.hr_option,
                "enabled": self.enabled,
            }
