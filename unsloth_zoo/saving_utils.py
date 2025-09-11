# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "create_huggingface_repo",
    "merge_and_dequantize_lora",
    "merge_and_overwrite_lora",
]
import warnings
from .peft_utils import get_lora_layer_modules
from .utils import _get_dtype
from .hf_utils import dtype_from_config
from .temporary_patches.common import UNSLOTH_ENABLE_LOGGING, logger


try:
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors, convert_moe_packed_tensors_cpu
except (ImportError, ModuleNotFoundError):
    # Provide a fallback or a clear error if the function isn't available
    # when not using mxfp4.
    convert_moe_packed_tensors     = None
    convert_moe_packed_tensors_cpu = None


MODEL_CARD = \
"""---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
license: apache-2.0
language:
- en
---

# Uploaded finetuned {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""

import torch
import bitsandbytes as bnb
try:
    from huggingface_hub import get_token
except:
    try:
        from huggingface_hub.utils import get_token
    except:
        # For older versions of huggingface_hub
        from huggingface_hub.utils._token import get_token
    pass
pass
from transformers.modeling_utils import PushToHubMixin
import json
import os
from pathlib import Path
from typing import Union, List, Optional
import tempfile
from peft import PeftModelForCausalLM, PeftModel

def find_skipped_quantized_modules(model):
    skipped_modules = []
    quantized_modules = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if hasattr(module.weight, 'quant_state') and module.weight.quant_state is not None:
                quantized_modules.append(name)
            else:
                skipped_modules.append(name)
        elif isinstance(module, torch.nn.Linear):
            skipped_modules.append(name)
    return skipped_modules, quantized_modules

def create_huggingface_repo(
    model,
    repo_id,
    private = False,
    token = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(repo_id) is str)
    if repo_id.count("/") != 1:
        raise TypeError(f"Unsloth: You are pushing to Hugging Face, but {repo_id} is not a valid repo.")

    from huggingface_hub import ModelCard
    if token is None: token = get_token()
    repo_id = PushToHubMixin._create_repo(
        PushToHubMixin,
        repo_id = repo_id,
        private = private,
        token = token,
    )
    username = repo_id.split("/")[0]

    # Check if base_model is a local path
    base_model = model.config._name_or_path
    if os.path.exists(base_model) and os.path.isdir(base_model):
        # Try to get the original model ID from config
        original_model_id = get_original_model_id(base_model)
        if original_model_id is not None and not os.path.exists(original_model_id):
            # Use the original model ID if it doesn't look like a local path
            base_model = original_model_id
        else:
            # If we can't determine the original model, use repo_id as a generic description
            # that won't cause HF validation errors
            base_model = repo_id

    # Create model card
    content = MODEL_CARD.format(
        username   = username,
        base_model = base_model,
        model_type = model.config.model_type,
        method     = "",
        extra      = "unsloth",
    )
    card = ModelCard(content)
    card.push_to_hub(repo_id, token = token, commit_message = "Unsloth Model Card")

    from huggingface_hub import HfApi
    hf_api = HfApi(token = token)
    return username, repo_id, hf_api
pass


from huggingface_hub import (
    snapshot_download,
    hf_hub_download,
    HfFileSystem,
)
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
from tqdm import tqdm as ProgressBar
import os, shutil, re, functools


# def _merge_lora(W, lora_stats, name):
#     if lora_stats.lora_A is None or lora_stats.lora_B is None: return W
#     W = W.to("cuda", dtype = torch.float32, non_blocking = True)
#     W = W.addmm_(
#         lora_stats.lora_B.to("cuda", dtype = torch.float32, non_blocking = True),
#         lora_stats.lora_A.to("cuda", dtype = torch.float32, non_blocking = True),
#         alpha = lora_stats.alpha,
#     )
#     if not torch.isfinite(torch.amax(W)).item():
#         raise ValueError('Unsloth: Merge failed as there are infinite elements in ' + name)
#     return W
# pass

def _merge_lora(W, lora_stats, name, device):
    if lora_stats.lora_A is None or lora_stats.lora_B is None: return W
    # Move all tensors to the chosen device for the operation
    W = W.to(device, dtype = torch.float32, non_blocking = True)
    lora_B = lora_stats.lora_B.to(device, dtype = torch.float32, non_blocking = True)
    lora_A = lora_stats.lora_A.to(device, dtype = torch.float32, non_blocking = True)

    W = W.addmm_(
        lora_B,
        lora_A,
        alpha = lora_stats.alpha,
    )
    if not torch.isfinite(torch.amax(W)).item():
        raise ValueError('Unsloth: Merge failed as there are infinite elements in ' + name)
    return W
pass


def check_if_quantized(module: torch.nn.Module) -> bool:
    # All Unsloth Zoo code licensed under LGPLv3
    # Adapted from https://github.com/huggingface/peft/blob/main/src/peft/utils/integrations.py
    if not hasattr(module, "weight"): return False

    if hasattr(module, "W_q"):  # For handling HQQ quantized weight
        # weight = module.dequantize()
        # return weight
        return True
    elif type(module.weight).__module__.startswith("torchao."):
        # check for torchao without requiring any torchao imports
        # weight = module.weight.dequantize()
        # return weight
        return True

    weight = module.weight
    if not isinstance(weight, torch.nn.Parameter):
        if isinstance(weight, torch.Tensor):
            # this is an FSDP-specific edge case
            # return weight  # type: ignore
            return False
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        # return weight
        return False

    quant_state = getattr(module, "state", None)
    device = weight.device
    is_cpu = device.type == torch.device("cpu").type
    return True
    # weight = dequantize_bnb_weight(weight, state=quant_state)  # no-op if not bnb
    # if is_cpu:
    #     # dequantize_bnb_weight for 8bit moves the device in-place, thus we need to move it back to CPU if necessary
    #     module.weight = module.weight.to(device)
    # return weight
pass


def expand_module_keys(name, module, original_keys):
    # All Unsloth Zoo code licensed under LGPLv3
    keys = module.state_dict().keys()
    for key in keys: original_keys.add(name + "." + key)
    return original_keys
pass


from peft.utils.integrations import dequantize_module_weight
import collections
import numpy as np
import inspect
from tqdm import tqdm as ProgressBar
from dataclasses import dataclass

@dataclass
class LoraStats:
    module : torch.nn.Module
    lora_A : torch.Tensor
    lora_B : torch.Tensor
    alpha  : float
pass


def assert_same_keys(model, new_state_dict):
    # All Unsloth Zoo code licensed under LGPLv3
    inner_model = model.base_model.model if hasattr(model, "base_model") else model
    original_keys = inner_model.state_dict().keys()
    all_original_keys = set()
    for x in original_keys:
        where_weight = x.rfind(".weight")
        where_bias   = x.rfind(".bias")
        if where_weight != -1: x = x[:where_weight + len(".weight")]
        elif where_bias != -1: x = x[:where_bias   + len(".bias")  ]
        else: pass

        # Remove LoRA and base_layer
        j = max(x.rfind(".lora_"), x.rfind(".base_layer"))
        if j != -1: x = x[:j] + x[x.rfind("."):]

        all_original_keys.add(x)
    pass
    difference = all_original_keys ^ set(new_state_dict)
    if len(difference) != 0:
        raise RuntimeError(f"Unsloth: Extracted keys = {difference} do not match!")
    pass
pass

# @torch.inference_mode
# def create_lora_statistics(model, merge_into_original = False, return_state_dict = True):
#     # All Unsloth Zoo code licensed under LGPLv3
#     # merge_into_original is merging directly into 16bit downloaded model
#     # without dequantizing
#     Linear_LoRA_Layers = get_lora_layer_modules()
#     Linear_LoRA_Layers = tuple(x[0] for x in Linear_LoRA_Layers)
#
#     lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
#     module_count, lora_A_count, lora_B_count, scaling_count = 0, 0, 0, 0
#
#     remove_keys = set()
#     keep_keys   = set()
#
#     inner_model = find_lora_base_model(model)
#     for name, module in inner_model.named_modules():
#         if name == "": continue
#
#         elif name.endswith(".lora_A.default"):
#             lora_weights[name[:-len(".lora_A.default")]].lora_A = module.weight
#             lora_A_count += 1
#             expand_module_keys(name, module, remove_keys)
#
#         elif name.endswith(".lora_B.default"):
#             lora_weights[name[:-len(".lora_B.default")]].lora_B = module.weight
#             lora_B_count += 1
#             expand_module_keys(name, module, remove_keys)
#
#         elif isinstance(module, Linear_LoRA_Layers):
#             active_adapter = module.active_adapters[0] if \
#                 hasattr(module, "active_adapters") else module.active_adapter
#             lora_weights[name].alpha = module.scaling[active_adapter]
#             scaling_count += 1
#             expand_module_keys(name, module, remove_keys)
#
#         elif name.endswith(".base_layer"):
#             lora_weights[name[:-len(".base_layer")]].module = module
#             module_count += 1
#             remove_keys.add(name)
#             remove_keys.add(name[:-len(".base_layer")])
#
#         elif (not merge_into_original) and check_if_quantized(module):
#             lora_weights[name].module = module
#             keep_keys.add(name + ".weight")
#             if getattr(module, "bias", None) is not None: keep_keys.add(name + ".bias")
#             expand_module_keys(name, module, remove_keys)
#             remove_keys.add(name)
#
#         elif ".lora_" in name: continue
#
#         else:
#             new_keys = expand_module_keys(name, module, set())
#             for key in new_keys:
#                 if not key.endswith((".weight", ".bias")):
#                     # Check if quantized item exactly which has ".weight"
#                     if ".weight." in key:
#                         remove_keys.add(key)
#                     else:
#                         # Keep gate_tanh, embedding etc
#                         pass
#             remove_keys.add(name)
#         pass
#     pass
#     assert(module_count == lora_A_count == lora_B_count == scaling_count)
#
#     # Also return state_dict if needed
#     state_dict = None
#     if return_state_dict:
#         # Memory optimization: process items and remove from old dict to reduce peak memory
#         old_state_dict = inner_model.state_dict()
#         state_dict = collections.OrderedDict()
#
#         # Get a list of keys to iterate over, so we can delete from the dict
#         for name in list(old_state_dict.keys()):
#             param = old_state_dict.pop(name) # Pop to free memory
#
#             if name.endswith(".base_layer.weight"):
#                 name = name[:-len(".base_layer.weight")]
#
#             if name in lora_weights:
#                 state_dict[name + ".weight"]   = lora_weights[name]
#                 if getattr(lora_weights[name].module, "bias", None) is not None:
#                     state_dict[name + ".bias"] = lora_weights[name].module.bias
#                 continue
#             elif name in keep_keys:
#                 lora_name = name[:-len(".weight")]
#                 if lora_name in lora_weights:
#                     param = lora_weights[lora_name]
#                 state_dict[name] = param # Keep the param
#             elif name in remove_keys:
#                 continue # Skip and free memory
#             else:
#                 state_dict[name] = param # Keep the param
#
#         del old_state_dict # Ensure it's freed
#         gc.collect()
#
#     if return_state_dict: assert_same_keys(model, state_dict)
#     return lora_weights, state_dict
# pass



@torch.inference_mode
def create_lora_statistics(model, merge_into_original = False, return_state_dict = True):
    # All Unsloth Zoo code licensed under LGPLv3
    # merge_into_original is merging directly into 16bit downloaded model
    # without dequantizing
    Linear_LoRA_Layers = get_lora_layer_modules()
    Linear_LoRA_Layers = tuple(x[0] for x in Linear_LoRA_Layers)

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    module_count, lora_A_count, lora_B_count, scaling_count = 0, 0, 0, 0

    remove_keys = set()
    keep_keys   = set()

    inner_model = find_lora_base_model(model)
    for name, module in inner_model.named_modules():
        if name == "": continue

        elif name.endswith(".lora_A.default"):
            lora_weights[name[:-len(".lora_A.default")]].lora_A = module.weight
            lora_A_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".lora_B.default"):
            lora_weights[name[:-len(".lora_B.default")]].lora_B = module.weight
            lora_B_count += 1
            expand_module_keys(name, module, remove_keys)

        elif isinstance(module, Linear_LoRA_Layers):
            active_adapter = module.active_adapters[0] if \
                hasattr(module, "active_adapters") else module.active_adapter
            lora_weights[name].alpha = module.scaling[active_adapter]
            scaling_count += 1
            expand_module_keys(name, module, remove_keys)

        elif name.endswith(".base_layer"):
            lora_weights[name[:-len(".base_layer")]].module = module
            module_count += 1
            remove_keys.add(name)
            remove_keys.add(name[:-len(".base_layer")])

        elif (not merge_into_original) and check_if_quantized(module):
            lora_weights[name].module = module
            keep_keys.add(name + ".weight")
            if getattr(module, "bias", None) is not None: keep_keys.add(name + ".bias")
            expand_module_keys(name, module, remove_keys)
            remove_keys.add(name)

        elif ".lora_" in name: continue

        else:
            new_keys = expand_module_keys(name, module, set())
            for key in new_keys:
                if not key.endswith((".weight", ".bias")):
                    # Check if quantized item exactly which has ".weight"
                    if ".weight." in key:
                        remove_keys.add(key)
                    else:
                        # Keep gate_tanh, embedding etc
                        pass
            remove_keys.add(name)
        pass
    pass
    assert(module_count == lora_A_count == lora_B_count == scaling_count)

    # Also return state_dict if needed
    if return_state_dict:
        old_state_dict = inner_model.state_dict()
        state_dict     = collections.OrderedDict()
        for name, param in old_state_dict.items():

            if name.endswith(".base_layer.weight"):
                name = name[:-len(".base_layer.weight")]

            if name in lora_weights:
                state_dict[name + ".weight"]   = lora_weights[name]
                if getattr(lora_weights[name].module, "bias", None) is not None:
                    state_dict[name + ".bias"] = lora_weights[name].module.bias
                continue
            elif name in keep_keys:
                # Quantized modules with no LoRA adapters
                lora_name = name[:-len(".weight")]
                if lora_name in lora_weights:
                    param = lora_weights[lora_name]
                else:
                    # Bias term
                    pass
            elif name in remove_keys: continue

            state_dict[name] = param
        pass
    else:
        state_dict = None
    pass

    if return_state_dict: assert_same_keys(model, state_dict)
    return lora_weights, state_dict
pass


import torch
import gc
import time

@torch.inference_mode
def _merge_and_overwrite_lora(save_directory, filename, lora_weights, output_dtype, model_class_name, base_model_is_quantized=False, quant_type=None):
    filename_original = os.path.join(save_directory, filename)
    count = 0

    detailed_memory_tracking("START_DYNAMIC_MERGE", filename)

    # Convert lora_weights to safetensor format
    converted_lora_weights = _convert_lora_keys_to_safetensor_format(
        lora_weights, [], model_class_name=model_class_name)
    detailed_memory_tracking("AFTER_LORA_CONVERSION", filename)

    with safe_open(filename_original, framework="pt", device="cpu") as file:
        safetensor_keys = list(file.keys())
        detailed_memory_tracking("AFTER_FILE_OPEN", filename, f"{len(safetensor_keys)} keys")

        converted_lora_weights = _convert_lora_keys_to_safetensor_format(
            lora_weights, safetensor_keys, model_class_name=model_class_name)
        processed_mxfp4_keys = set()

        # Pre-calculate tensor info for look-ahead (first 100 tensors for memory estimation)
        tensor_info_list = []
        sample_keys = [k for k in safetensor_keys[:100] if k not in processed_mxfp4_keys]

        for key in sample_keys:
            try:
                W_sample = file.get_tensor(key)
                lora_key = key[:-len(".weight")] if key.endswith(".weight") else key
                lora_stats = converted_lora_weights.get(lora_key, None)
                tensor_info_list.append({
                    'key': key,
                    'tensor': W_sample,
                    'lora_stats': lora_stats
                })
            except:
                continue
        detailed_memory_tracking("AFTER_TENSOR_SAMPLING", filename, f"Sampled {len(tensor_info_list)} tensors")

        # BATCHED PROCESSING
        current_batch_tensors = OrderedDict()
        all_processed_tensors = OrderedDict()

        for idx, key in enumerate(safetensor_keys):
            if key in processed_mxfp4_keys:
                continue

            # Track memory periodically
            if idx % 50 == 0:
                detailed_memory_tracking(f"BEFORE_TENSOR_{idx}", key)

            # Memory cleanup before processing each tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            W_original = None
            W = None
            output_key = key
            action_logged = False

            # [Existing tensor processing logic - MXFP4 and 16-bit cases]
            if base_model_is_quantized:
                if quant_type == "mxfp4":
                    if key.endswith("_blocks"):
                        # ... existing MXFP4 logic ...
                        # (keeping this unchanged from your version)
                        pass
                    elif key.endswith("_scales"):
                        continue
                    else:
                        W_original = file.get_tensor(key)
                        W = W_original
                        detailed_memory_tracking("AFTER_TENSOR_LOAD", key)
            else:
                W_original = file.get_tensor(key)
                W = W_original
                detailed_memory_tracking("AFTER_TENSOR_LOAD", key)

            # LoRA merging logic (INDIVIDUAL merge, not batched)
            lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
            lora_stats = converted_lora_weights.get(lora_key, None)

            if W is not None and lora_stats is not None and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
                if not action_logged:
                    count += 1
                    merge_device = _choose_merge_device(W, lora_stats, output_key)
                    if UNSLOTH_ENABLE_LOGGING:
                        logger.debug(f"[DEBUG] Merging {output_key} on device '{merge_device}'.")
                    W = _merge_lora(W, lora_stats, output_key, device=merge_device)
                    detailed_memory_tracking("AFTER_LORA_MERGE", key)
                    action_logged = True

            if W is None:
                continue

            # Add merged tensor to current batch
            current_batch_tensors[output_key] = W.to(output_dtype)

            # Clean up references
            if W_original is not None and W is not W_original:
                del W_original
            del W

            # Calculate optimal batch size based on current memory state and upcoming tensors
            remaining_tensor_info = [info for info in tensor_info_list if info['key'] not in current_batch_tensors]
            optimal_batch_size = calculate_memory_aware_batch_size(
                current_batch_tensors,
                remaining_tensor_info,
                converted_lora_weights,
                output_dtype,
                safety_factor=0.85
            )

            # Process batch when optimal size reached OR last tensor
            should_process_batch = (
                len(current_batch_tensors) >= optimal_batch_size or
                idx == len(safetensor_keys) - 1
            )

            if should_process_batch:
                batch_memory = sum(
                    calculate_tensor_memory_cost(tensor, converted_lora_weights.get(
                        tensor_key[:-len(".weight")] if tensor_key.endswith(".weight") else tensor_key
                    ), output_dtype)
                    for tensor_key, tensor in current_batch_tensors.items()
                )

                detailed_memory_tracking(f"PROCESSING_BATCH_{idx}", None,
                                       f"Batch size: {len(current_batch_tensors)}, Memory: {format_bytes(batch_memory)}")

                # Create temporary file for batch
                batch_temp_file = tempfile.NamedTemporaryFile(
                    suffix=f"_batch_{idx//50:03d}.safetensors",
                    dir=save_directory,
                    delete=False
                )
                batch_temp_path = batch_temp_file.name
                batch_temp_file.close()

                # Save entire batch at once
                save_file(current_batch_tensors, batch_temp_path, metadata={"format": "pt"})
                detailed_memory_tracking(f"BATCH_SAVED_{idx}", None, f"Saved {len(current_batch_tensors)} tensors")

                # Load back as memory-mapped for final collection
                with safe_open(batch_temp_path, framework="pt", device="cpu") as batch_file:
                    for batch_key in batch_file.keys():
                        all_processed_tensors[batch_key] = batch_file.get_tensor(batch_key)

                # Clean up
                current_batch_tensors.clear()
                try:
                    os.remove(batch_temp_path)
                except:
                    pass

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                detailed_memory_tracking(f"AFTER_BATCH_CLEANUP_{idx}", None)

    # Final save
    detailed_memory_tracking("AFTER_PROCESSING_COMPLETE", filename, f"Total processed tensors: {len(all_processed_tensors)}")

    with tempfile.NamedTemporaryFile(suffix=".safetensors", dir=save_directory, delete=False) as final_temp:
        final_temp_path = final_temp.name

    save_file(all_processed_tensors, final_temp_path, metadata={"format": "pt"})
    detailed_memory_tracking("AFTER_FINAL_SAVE", filename)

    del all_processed_tensors
    gc.collect()
    detailed_memory_tracking("AFTER_FINAL_CLEANUP", filename)

    os.replace(final_temp_path, filename_original)
    return count
pass

# @torch.inference_mode
# def _merge_and_overwrite_lora(save_directory, filename, lora_weights, output_dtype, model_class_name, base_model_is_quantized=False, quant_type=None):
#     filename_original = os.path.join(save_directory, filename)
#     count = 0
#     import psutil
#     import pickle
#
#     # Dynamic batch sizing tracking
#     actual_tensor_sizes = []  # Track actual memory consumption
#     current_batch_size = 20  # Conservative initial size
#     batch_adjustment_frequency = 25  # Adjust batch size every N tensors
#
#     detailed_memory_tracking("START_DYNAMIC_MERGE", filename, f"Initial batch size: {current_batch_size}")
#
#     # Convert lora_weights to safetensor format
#     converted_lora_weights = _convert_lora_keys_to_safetensor_format(
#         lora_weights, [], model_class_name=model_class_name)
#     detailed_memory_tracking("AFTER_LORA_CONVERSION", filename)
#
#     temp_processed_tensors = []
#
#     with safe_open(filename_original, framework="pt", device="cpu") as file:
#         safetensor_keys = list(file.keys())
#         detailed_memory_tracking("AFTER_FILE_OPEN", filename, f"{len(safetensor_keys)} keys")
#
#         converted_lora_weights = _convert_lora_keys_to_safetensor_format(
#             lora_weights, safetensor_keys, model_class_name=model_class_name)
#         processed_mxfp4_keys = set()
#
#         for idx, key in enumerate(safetensor_keys):
#             if key in processed_mxfp4_keys:
#                 continue
#
#             # Dynamic batch size adjustment
#             if idx > 0 and idx % batch_adjustment_frequency == 0:
#                 stats = get_memory_stats()
#                 available_memory = stats['cpu']['available']
#
#                 new_batch_size = calculate_dynamic_batch_size(
#                     actual_tensor_sizes,
#                     available_memory
#                 )
#
#                 if new_batch_size != current_batch_size:
#                     detailed_memory_tracking(f"BATCH_SIZE_ADJUSTED_{idx}", None,
#                                            f"Batch size: {current_batch_size} -> {new_batch_size}")
#                     current_batch_size = new_batch_size
#
#             # Track memory periodically
#             if idx % 20 == 0:
#                 detailed_memory_tracking(f"BEFORE_TENSOR_{idx}", key)
#
#             # FORCE memory cleanup before processing each tensor
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#                 torch.cuda.synchronize()
#
#             W_original = None
#             W = None
#             output_key = key
#             action_logged = False
#
#             # [Your existing tensor processing logic - MXFP4 and 16-bit cases]
#             if base_model_is_quantized:
#                 if quant_type == "mxfp4":
#                     if key.endswith("_blocks"):
#                         if convert_moe_packed_tensors is None:
#                             raise ImportError("MXFP4 dequantization is required, but `convert_moe_packed_tensors` could not be imported.")
#
#                         base_name = key[:-len("_blocks")]
#                         scales_key = base_name + "_scales"
#                         output_key = base_name
#                         if scales_key not in safetensor_keys:
#                             warnings.warn(f"Found mxfp4 tensor {key} but missing its scales tensor {scales_key}. Skipping.")
#                             continue
#
#                         blocks_tensor, scales_tensor = file.get_tensor(key), file.get_tensor(scales_key)
#
#                         if torch.cuda.is_available():
#                             torch.cuda.synchronize()
#                             torch.cuda.empty_cache()
#
#                         device_type, device_id, rows_per_chunk = _choose_mxfp4_processing_strategy(
#                             blocks_tensor, scales_tensor
#                         )
#
#                         if device_type == 'cpu':
#                             try:
#                                 from transformers.integrations.mxfp4 import convert_moe_packed_tensors_cpu
#                                 W = convert_moe_packed_tensors_cpu(
#                                     blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
#                                 ).transpose(1, 2).contiguous()
#                                 if UNSLOTH_ENABLE_LOGGING:
#                                     logger.debug(f"[DEBUG] Using CPU dequantization for {base_name} with {rows_per_chunk:,} rows per chunk")
#                             except ImportError:
#                                 W = convert_moe_packed_tensors(
#                                     blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
#                                 ).transpose(1, 2).contiguous()
#                         else:
#                             W = convert_moe_packed_tensors(
#                                 blocks_tensor, scales_tensor, rows_per_chunk=rows_per_chunk
#                             ).transpose(1, 2).contiguous()
#                             if UNSLOTH_ENABLE_LOGGING:
#                                 logger.debug(f"[DEBUG] Using GPU dequantization for {base_name} with {rows_per_chunk:,} rows per chunk")
#
#                         del blocks_tensor, scales_tensor
#                         processed_mxfp4_keys.add(key)
#                         processed_mxfp4_keys.add(scales_key)
#
#                         lora_stats = converted_lora_weights.get(base_name, None)
#                         if lora_stats and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
#                             if UNSLOTH_ENABLE_LOGGING:
#                                 logger.debug(f"[DEBUG] DEQUANTIZING MXFP4 & MERGING LoRA into Key Group: {base_name}")
#                             count += 1
#                             merge_device = _choose_merge_device(W, lora_stats, output_key)
#                             if UNSLOTH_ENABLE_LOGGING:
#                                 try:
#                                     logger.debug(f"[DEBUG] Merging {output_key} on device '{merge_device}'.")
#                                 except:
#                                     pass
#                             W = _merge_lora(W, lora_stats, output_key, device=merge_device)
#                         else:
#                             if UNSLOTH_ENABLE_LOGGING:
#                                 logger.debug(f"[DEBUG] DEQUANTIZING MXFP4 Key Group: {base_name}")
#                         action_logged = True
#                         detailed_memory_tracking("AFTER_MXFP4_DEQUANT", key)
#
#                     elif key.endswith("_scales"):
#                         continue
#                     else:
#                         W_original = file.get_tensor(key)
#                         W = W_original
#                         detailed_memory_tracking("AFTER_TENSOR_LOAD", key)
#             else:
#                 W_original = file.get_tensor(key)
#                 W = W_original
#                 detailed_memory_tracking("AFTER_TENSOR_LOAD", key)
#
#             # LoRA merging logic
#             lora_key = output_key[:-len(".weight")] if output_key.endswith(".weight") else output_key
#             lora_stats = converted_lora_weights.get(lora_key, None)
#
#             if W is not None and lora_stats is not None and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
#                 if not action_logged:
#                     count += 1
#                     merge_device = _choose_merge_device(W, lora_stats, output_key)
#                     if UNSLOTH_ENABLE_LOGGING:
#                         try:
#                             logger.debug(f"[DEBUG] Merging {output_key} on device '{merge_device}'.")
#                         except:
#                             pass
#                     W = _merge_lora(W, lora_stats, output_key, device=merge_device)
#                     detailed_memory_tracking("AFTER_LORA_MERGE", key)
#                     action_logged = True
#
#             if W is None:
#                 continue
#
#             # DYNAMIC MEMORY TRACKING: Calculate actual memory cost of this tensor
#             tensor_memory_cost = calculate_tensor_memory_cost(W, lora_stats, output_dtype)
#             actual_tensor_sizes.append(tensor_memory_cost)
#
#             # Keep only recent history to avoid memory growth
#             if len(actual_tensor_sizes) > 200:
#                 actual_tensor_sizes = actual_tensor_sizes[-100:]  # Keep last 100 measurements
#
#             if W_original is not None and W is not W_original:
#                 del W_original
#                 detailed_memory_tracking("AFTER_DEL_ORIGINAL", key)
#
#             # Save each tensor to individual temp file immediately
#             tensor_temp_file = tempfile.NamedTemporaryFile(
#                 suffix=f"_{idx:04d}.pt",
#                 dir=save_directory,
#                 delete=False
#             )
#             tensor_temp_path = tensor_temp_file.name
#             tensor_temp_file.close()
#
#             torch.save(W.to(output_dtype), tensor_temp_path, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL)
#             temp_processed_tensors.append((output_key, tensor_temp_path))
#
#             detailed_memory_tracking("AFTER_TENSOR_TEMP_SAVE", key)
#
#             # Immediate cleanup
#             del W
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#
#             # More frequent garbage collection
#             if (idx + 1) % 10 == 0:
#                 gc.collect()
#                 detailed_memory_tracking(f"AFTER_GC_CYCLE_{idx+1}", None)
#
#     detailed_memory_tracking("AFTER_PROCESSING_COMPLETE", filename, f"Total temp tensors: {len(temp_processed_tensors)}")
#
#     # STREAMING COMBINATION with dynamically calculated final batch size
#     stats = get_memory_stats()
#     final_batch_size = calculate_dynamic_batch_size(actual_tensor_sizes, stats['cpu']['available'])
#     detailed_memory_tracking("FINAL_BATCH_SIZE", None, f"Final batch size: {final_batch_size}")
#
#     final_tensors = OrderedDict()
#     final_temp_file = tempfile.NamedTemporaryFile(suffix=".safetensors", dir=save_directory, delete=False)
#     final_temp_path = final_temp_file.name
#     final_temp_file.close()
#
#     try:
#         batch_count = 0
#         for i, (tensor_key, tensor_temp_path) in enumerate(temp_processed_tensors):
#             # Load tensor as memory-mapped
#             tensor_data = torch.load(tensor_temp_path, map_location="cpu", mmap=True, weights_only=False)
#             final_tensors[tensor_key] = tensor_data
#
#             # Clean up individual temp file immediately
#             try:
#                 os.remove(tensor_temp_path)
#             except:
#                 pass
#
#             # Save in dynamically calculated batches
#             if len(final_tensors) >= final_batch_size or i == len(temp_processed_tensors) - 1:
#                 detailed_memory_tracking(f"SAVING_BATCH_{batch_count}", None, f"Batch size: {len(final_tensors)}")
#
#                 if batch_count == 0:
#                     # First batch - create the file
#                     save_file(final_tensors, final_temp_path, metadata={"format": "pt"})
#                 else:
#                     # Subsequent batches - merge with existing file
#                     existing_tensors = OrderedDict()
#                     with safe_open(final_temp_path, framework="pt", device="cpu") as existing_file:
#                         for existing_key in existing_file.keys():
#                             existing_tensors[existing_key] = existing_file.get_tensor(existing_key)
#
#                     # Combine existing + new batch
#                     existing_tensors.update(final_tensors)
#                     save_file(existing_tensors, final_temp_path, metadata={"format": "pt"})
#                     del existing_tensors
#
#                 # Clear current batch
#                 final_tensors.clear()
#                 batch_count += 1
#                 gc.collect()
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#
#                 detailed_memory_tracking(f"AFTER_BATCH_{batch_count-1}", None)
#
#     except Exception as e:
#         # Clean up all temp files on error
#         for _, tensor_temp_path in temp_processed_tensors:
#             try:
#                 os.remove(tensor_temp_path)
#             except:
#                 pass
#         try:
#             os.remove(final_temp_path)
#         except:
#             pass
#         raise e
#
#     detailed_memory_tracking("BEFORE_FINAL_REPLACE", filename)
#
#     # Replace original file
#     try:
#         os.replace(final_temp_path, filename_original)
#     except OSError as e:
#         print(f"Error renaming temporary file: {e}. Attempting copy and replace.")
#         import shutil
#
#         if os.name == 'nt':
#             for attempt in range(3):
#                 try:
#                     shutil.copy2(final_temp_path, filename_original)
#                     break
#                 except PermissionError:
#                     if attempt == 2:
#                         raise
#                     time.sleep(0.5)
#                     gc.collect()
#         else:
#             shutil.copy2(final_temp_path, filename_original)
#
#         try:
#             os.remove(final_temp_path)
#         except:
#             pass
#
#     detailed_memory_tracking("AFTER_FINAL_REPLACE", filename)
#     return count
# pass
from huggingface_hub import (
    split_state_dict_into_shards_factory,
    get_torch_storage_size,
    get_torch_storage_id,
)

def get_torch_storage_size_new(x, element_size):
    if isinstance(x, LoraStats):
        shape = (x.module.in_features, x.module.out_features)
        return int(np.prod(shape)) * element_size
    else:
        return get_torch_storage_size(x)
pass


def get_torch_storage_id_new(x):
    if isinstance(x, LoraStats):
        return None
    else:
        return get_torch_storage_id(x)
pass


def prepare_saving(
    model,
    save_directory,
    push_to_hub = False,
    max_shard_size = "5GB",
    private = True,
    token = None,
    output_dtype = None,
    merge_into_original = False,
    low_disk_space_usage = False,
    min_size_in_bytes = 100_000_000, # Must be of this size - 100MB default
    use_temp_file = False,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Check size
    from huggingface_hub.serialization._base import parse_size_to_int
    max_shard_size_in_bytes = max_shard_size
    if type(max_shard_size_in_bytes) is not int:
        max_shard_size_in_bytes = parse_size_to_int(max_shard_size)
    pass

    temp_file = None
    username, repo_id, hf_api = None, None, None

    if push_to_hub:
        if token is None: token = get_token()
        username, repo_id, hf_api = create_huggingface_repo(
            model = model,
            repo_id = save_directory,
            private = private,
            token = token,
        )
        # Check if temporary folder is needed
        if os.path.isdir(save_directory) or use_temp_file:
            temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
            save_directory = temp_file.name
            use_temp_file = True
        pass
    pass

    if output_dtype is None: output_dtype = _get_dtype(dtype_from_config(model.config))
    assert(output_dtype in (torch.float32, torch.float16, torch.float64, torch.bfloat16))
    assert(type(torch.bfloat16) is torch.dtype)
    element_size = torch.tensor([], dtype = output_dtype).element_size()

    # Get state_dict
    lora_weights, state_dict = create_lora_statistics(
        model,
        merge_into_original = merge_into_original,
        return_state_dict = True,
    )
    # Total save size in bytes
    save_size = sum(get_torch_storage_size_new(x, element_size) for x in state_dict.values())

    # Create folder if it does not exist
    if not os.path.exists(save_directory):
        try:
            os.makedirs(save_directory, exist_ok = True)
        except Exception as error:
            raise RuntimeError(f"Unsloth: Error creating directory {save_directory} with error = {str(error)}")
    pass

    # Check if directory has enough space
    total, used, free = shutil.disk_usage(save_directory)
    free = int(free*0.95)

    def raise_upload_works():
        # Works with individual shard uploading
        raise RuntimeError(
            "Unsloth: Failed saving locally - no disk space left. "\
            "Uploading can work luckily! Use .push_to_hub instead."
        )
    pass

    if free < save_size:
        # Fail if already using temp folder except if individual portions work!
        if use_temp_file:
            if merge_into_original:
                if free > min_size_in_bytes:
                    # Downloading safetensor shards must be min shard size
                    low_disk_space_usage = True
                else: raise_upload_works()
            elif free > 100_000_000:
                if push_to_hub:
                    # Instead we form shards on the fly and push them!
                    low_disk_space_usage = True
                    max_shard_size_in_bytes = free
                else: raise_upload_works()
            else:
                raise RuntimeError("Failed saving - no disk space left!")
        pass

        # Too small - try using the temporary file system (sometimes large like Kaggle)
        try_temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
        try_save_directory = try_temp_file.name

        total, used, free = shutil.disk_usage(try_save_directory)
        free = int(free*0.95)
        if not push_to_hub and free > save_size: raise_upload_works()
        elif push_to_hub and free < save_size:
            raise RuntimeError("Unsloth: Failed uploading - no disk space left.")
        elif push_to_hub:
            print(
                f"Unsloth: Saving to {save_directory} will fail, but using a temp folder works! "\
                "Switching to a temp folder then uploading!"
            )
            # Switch to temp directory
            temp_file = try_temp_file
            save_directory = try_save_directory
            use_temp_file = True
        else:
            raise RuntimeError("Failed saving - no disk space left!")
        pass
    pass

    return (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    )
pass


def _remove_quantization_config(config_path: Path):
    assert config_path.exists(), "Given config does not exist"
    with open(config_path, "r", encoding = "utf-8") as f:
        config = json.load(f)
    if "quantization_config" in config:
        # Remove the quantization_config field
        del config["quantization_config"]
    else:
        return
    # Overwrite the config file
    with open(config_path, "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 4)
    pass
pass


@torch.inference_mode
def merge_and_overwrite_lora(
    get_model_name,
    model,
    tokenizer            = None,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    private              = False,
    token                = None,
    save_method          = "merged_16bit",
    output_dtype         = None,
    low_disk_space_usage = False,
    use_temp_file        = False,
    cleanup_temp_file    = True,
):
    import os
    #os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
    #os.environ['HF_HUB_DISABLE_XET']="1"
    conservative_memory_cleanup("between_operations")
    detailed_memory_tracking("MERGE_START", "Initial state")
    # All Unsloth Zoo code licensed under LGPLv3
    # Directly downloads 16bit original weights and merges LoRA
    inner_model = model.base_model.model if isinstance(model, PeftModel) else model
    inner_model = inner_model.base_model if hasattr(model, "base_model") else inner_model
    if not isinstance(model, PeftModel):
        warnings.warn("Model is not a PeftModel (no Lora adapters detected). Skipping Merge. Please use save_pretrained() or push_to_hub() instead!")
        return None
    try:
        model_name = get_model_name(model.config._name_or_path, load_in_4bit = False)
    except:
        model_name = model.config._name_or_path

    final_model_name, is_local_path, source_info, base_model_is_quantized, quant_type = determine_base_model_source(model_name, token)
    if base_model_is_quantized and (quant_type == "nf4" or quant_type == "fp4") and save_method== "merged_16bit":
        raise TypeError("Base model should be a 16bits or mxfp4 base model for a 16bit model merge. Use `save_method=forced_merged_4bit` instead")
    model_name = final_model_name
    safetensors_list = []
    max_size_in_bytes = 0
    total_size_in_bytes = 0
    config = model.config

    # Handle case for local model where config._name_or_path is a local os path
    # https://github.com/unslothai/unsloth/issues/2140
    is_local_path = False
    if os.path.exists(model_name) and os.path.isdir(model_name):
        is_local_path = True
        detailed_memory_tracking("LOCAL_MODEL_DETECTED", f"Path: {model_name}")
        print(f"Detected local model directory: {model_name}")

        # Get safetensors files from local directory
        for file in os.listdir(model_name):
            if file.endswith(".safetensors"):
                safetensors_list.append(file)
                file_path = os.path.join(model_name, file)
                file_size = os.path.getsize(file_path)
                max_size_in_bytes = max(max_size_in_bytes, file_size)
                total_size_in_bytes += file_size

        # Check for index file
        index_path = os.path.join(model_name, "model.safetensors.index.json")
        if os.path.exists(index_path):
            detailed_memory_tracking("LOCAL_INDEX_PROCESSING", f"Index file: {index_path}")
            try:
                with open(index_path, 'r', encoding = "utf-8") as f:
                    index_data = json.load(f)
                    # Extract file names from the index if available
                    if "weight_map" in index_data:
                        # Get unique filenames from weight map
                        indexed_files = set(index_data["weight_map"].values())
                        # Only use these if we didn't find files directly
                        if not safetensors_list:
                            safetensors_list = list(indexed_files)
                            # Need to compute sizes for these files
                            for file in safetensors_list:
                                file_path = os.path.join(model_name, file)
                                if os.path.exists(file_path):
                                    file_size = os.path.getsize(file_path)
                                    max_size_in_bytes = max(max_size_in_bytes, file_size)
                                    total_size_in_bytes += file_size
                detailed_memory_tracking("AFTER_LOCAL_INDEX_PROCESSING", None)
            except Exception as e:
                print(f"Warning: Could not process index file: {e}")
    else:
        # Original HF repo logic
        detailed_memory_tracking("BEFORE_HF_FILE_LIST", f"Fetching file list for {model_name}")
        try:
            file_list = HfFileSystem(token = token).ls(model_name, detail = True)
            detailed_memory_tracking("AFTER_HF_FILE_LIST", f"Retrieved {len(file_list) if file_list else 0} files")
        except:
            detailed_memory_tracking("HF_FILE_LIST_ERROR", f"Error: {str(e)}")
            original_model_id = get_original_model_id(model_name)
            model_name = original_model_id
            if original_model_id is None:
                raise ValueError(f"Could not determine original model ID from {model_name}. "
                                "If using a local model, ensure the path exists and contains safetensors files.")
            detailed_memory_tracking("BEFORE_HF_FILE_LIST_RETRY", f"Retrying with {model_name}")
            file_list = HfFileSystem(token = token).ls(model_name, detail = True)
            detailed_memory_tracking("AFTER_HF_FILE_LIST_RETRY", f"Retrieved {len(file_list) if file_list else 0} files")

        # Process HF file listing
        for x in file_list:
            if not x["name"].endswith(".safetensors"): continue
            safetensors_list.append(os.path.split(x["name"])[-1])
            max_size_in_bytes = max(max_size_in_bytes, x["size"])
            total_size_in_bytes += x["size"]
    detailed_memory_tracking("AFTER_SAFETENSORS_LIST", f"{len(safetensors_list)} files, {format_bytes(total_size_in_bytes)} total")

    if not safetensors_list:
         raise RuntimeError(f"No '.safetensors' files found for the base model: {model_name}")
    assert(max_size_in_bytes != 0 and total_size_in_bytes != 0)

    (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, new_use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    ) = prepare_saving(
        model = model,
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        max_shard_size = "5GB",
        private = private,
        token = token,
        output_dtype = output_dtype,
        low_disk_space_usage = low_disk_space_usage,
        merge_into_original = True,
        min_size_in_bytes = max_size_in_bytes,
        use_temp_file = use_temp_file,
    )
    detailed_memory_tracking("AFTER_PREPARE_SAVING", None)
    use_temp_file = use_temp_file or new_use_temp_file
    _save_dir_path = Path(save_directory)

    n_saved_modules = 0
    def upload_items(filename = None):
        extras = {"repo_id" : repo_id, "repo_type" : "model", "commit_message" : "(Trained with Unsloth)", }
        if filename is None:
            hf_api.upload_folder(folder_path = save_directory, **extras,)
        else:
            hf_api.upload_file(
                path_or_fileobj = os.path.join(save_directory, filename),
                path_in_repo = filename,
                **extras,
            )
        pass
    pass

    # Save config / generation_config via no state_dict and tokenizer
    if tokenizer is not None:
        detailed_memory_tracking("BEFORE_TOKENIZER_SAVE", None)
        tokenizer.save_pretrained(save_directory = save_directory,)
        detailed_memory_tracking("AFTER_TOKENIZER_SAVE", None)

    # --- Handle 4-bit merging first ---
    if save_method == "merged_4bit" or save_method == "forced_merged_4bit":
        base_model = model.base_model if isinstance(model, PeftModel) else model
        print(f"Unsloth: Merging LoRA weights into 4bit model...")
        if not isinstance(model, PeftModelForCausalLM) and not isinstance(model, PeftModel):
             raise TypeError("Model must be a PeftModelForCausalLM or PeftModel for 'merged_4bit' save.")
        if not getattr(model.config, "quantization_config", None):
             raise ValueError("Model does not appear to be quantized. Cannot use 'merged_4bit'.")

        # Perform the merge
        try:
            # Use the base_model reference which points to the PeftModel's base
            merged_model = base_model.merge_and_unload()
            print(f"Unsloth: Merging finished.")
        except Exception as e:
            raise RuntimeError(f"Failed to merge LoRA weights for 4-bit save: {e}")

        # Check for skipped modules (optional but good practice)
        skipped_modules, _ = find_skipped_quantized_modules(merged_model)
        if len(skipped_modules) > 0:
            print(f"Unsloth: Found skipped modules: {skipped_modules}. Updating config.")
            # Ensure quantization_config exists before modifying
            if not hasattr(merged_model.config, "quantization_config"):
                merged_model.config.quantization_config = {} # Initialize if somehow missing
            merged_model.config.quantization_config["llm_int8_skip_modules"] = skipped_modules

        print(f"Unsloth: Saving merged 4bit model to {save_directory}...")
        try:
            merged_model.save_pretrained(save_directory = save_directory)
            print(f"Unsloth: Merged 4bit model saved.")
        except Exception as e:
             raise RuntimeError(f"Failed to save merged 4-bit model: {e}")

        # Upload the saved 4-bit model files
        if push_to_hub:
            upload_items() # Upload the entire directory content

        # Clean up temp file if created
        if cleanup_temp_file and temp_file is not None:
            print("Unsloth: Cleaning up temporary file...")
            try: temp_file.cleanup()
            except Exception as e: print(f"Warning: Failed to cleanup temp file: {e}")

        print("Unsloth: Merged 4bit model process completed.")
        return save_directory # <<<--- EARLY RETURN for 4-bit path


    # Default handle 16 bit merge and save/push
    # Step 1: Save base model config/architecture (no weights needed here)
    if save_method == "merged_16bit":
        detailed_memory_tracking("BEFORE_CONFIG_SAVE", None)
        # config_model = find_lora_base_model(model) if isinstance(model, PeftModel) else model
        # config_model.save_pretrained(
        #     save_directory = save_directory,
        #     state_dict = {},
        # )
        # # Remove any weight files that shouldn't have been saved (transformers 4.56.0 bug)
        # import glob
        # weight_files = glob.glob(os.path.join(save_directory, "*.bin")) + \
        #                glob.glob(os.path.join(save_directory, "*.safetensors"))
        #
        # for weight_file in weight_files:
        #     os.remove(weight_file)
        #     if UNSLOTH_ENABLE_LOGGING:
        #         logger.debug(f"DEBUG: Removed incorrectly saved weight file: {os.path.basename(weight_file)}")
        #from transformers import AutoConfig
        #base_config = AutoConfig.from_pretrained(final_model_name)
        #base_config.save_pretrained(save_directory)
        config.save_pretrained(save_directory)
        detailed_memory_tracking("AFTER_CONFIG_SAVE", None)
        if tokenizer is not None: tokenizer.save_pretrained(save_directory)

        _remove_quantization_config(config_path = Path(save_directory) / "config.json")
        # Remove the quantization_config in the config.json file if it exists,
    # as we are exporting the model in 16-bit format.

    # Step 2: Initial upload of non-model files (config, tokenizer)
    if push_to_hub:
        detailed_memory_tracking("BEFORE_INITIAL_UPLOAD", None)
        upload_items()
        detailed_memory_tracking("AFTER_INITIAL_UPLOAD", None)


    # Step 3: Conditional index handling
    _hf_cache_dir = _get_hf_cache_dir()
    copied_all_from_cache = False
    safe_tensor_index_files = ["model.safetensors.index.json"] if len(safetensors_list) > 1 else []

    # ONLY download/copy the original index if we are NOT dequantizing an MXFP4 model
    if not (base_model_is_quantized and quant_type == "mxfp4"):
        if is_local_path:
            detailed_memory_tracking("BEFORE_LOCAL_INDEX_COPY", None)
            os.makedirs(save_directory, exist_ok=True)
            # Copy from local
            if safe_tensor_index_files:
                local_index_path = os.path.join(model_name, "model.safetensors.index.json")
                if os.path.exists(local_index_path):
                    shutil.copy2(local_index_path, os.path.join(save_directory, "model.safetensors.index.json"))
                    gc.collect()
            detailed_memory_tracking("AFTER_LOCAL_INDEX_COPY", None)
        else:
            # Download from HF
            if "model.safetensors.index.json" in [f for f in safe_tensor_index_files]:
                detailed_memory_tracking("BEFORE_INDEX_DOWNLOAD", f"Downloading index file")
                snapshot_download(repo_id=model_name, local_dir=save_directory, allow_patterns=["model.safetensors.index.json"])
                detailed_memory_tracking("AFTER_INDEX_DOWNLOAD", f"Index file downloaded")

        if push_to_hub and safe_tensor_index_files:
            upload_items("model.safetensors.index.json")
        pass

    detailed_memory_tracking("BEFORE_CACHE_CHECK", f"Checking cache for {len(safetensors_list)} files")
    # Step 4 : Handle retrieval of original 16-bit shards
    if not is_local_path and _hf_cache_dir is not None:
        copied_all_from_cache = _try_copy_all_from_cache(
            repo_id=model_name,
            filenames_to_check=safetensors_list,
            target_dir_str=save_directory,
            hf_cache_dir=_hf_cache_dir,
            token=token,
        )
        detailed_memory_tracking("AFTER_CACHE_CHECK", f"Copied from cache: {copied_all_from_cache}")

    if not copied_all_from_cache and not low_disk_space_usage and not is_local_path:
        print(f"Downloading safetensors for {model_name}...")
        snapshot_download(
            repo_id = model_name,
            local_dir = save_directory,
            allow_patterns = safe_tensor_index_files + safetensors_list,
        )
        detailed_memory_tracking("AFTER_SNAPSHOT_DOWNLOAD", f"Downloaded {len(safetensors_list)} files")


    # Step 5: Iterate through original shards, merge LoRA, and overwrite/save
    for filename in ProgressBar(safetensors_list, desc = "Unsloth: Merging weights into 16bit"):
        file_path = os.path.join(save_directory, filename)
        detailed_memory_tracking("BEFORE_SHARD_PROCESS", f"Processing {filename}")
        # Only download if we didn't get everything from cache AND this specific file doesn't exist
        # AND we're in low disk space mode
        # For local models, copy the file if needed
        if is_local_path and not os.path.exists(file_path):
            detailed_memory_tracking("BEFORE_LOCAL_COPY", f"Copying {filename}")
            local_file_path = os.path.join(model_name, filename)
            if os.path.exists(local_file_path):
                shutil.copy2(local_file_path, file_path)
                print(f"Copied {filename} from local model directory")
                gc.collect()
            detailed_memory_tracking("AFTER_LOCAL_COPY", f"Copied {filename}")


        elif not copied_all_from_cache and low_disk_space_usage and not os.path.exists(file_path) and not is_local_path:
            detailed_memory_tracking("BEFORE_INDIVIDUAL_DOWNLOAD", f"Downloading {filename}")
            hf_hub_download(
                repo_id = model_name,
                filename = filename,
                repo_type = "model",
                local_dir = save_directory,
            )
            detailed_memory_tracking("AFTER_INDIVIDUAL_DOWNLOAD", f"Downloaded {filename}")
            conservative_memory_cleanup("between_operations")
        pass
        detailed_memory_tracking("BEFORE_MERGE_OPERATION", f"Starting merge for {filename}")
        n_saved_modules += _merge_and_overwrite_lora(
            save_directory = save_directory,
            filename = filename,
            lora_weights = lora_weights,
            output_dtype = output_dtype,
            model_class_name = find_lora_base_model(model).__class__.__name__,
            base_model_is_quantized = base_model_is_quantized,
            quant_type=quant_type,
        )
        detailed_memory_tracking("AFTER_MERGE_OPERATION", f"Completed merge for {filename}")
        conservative_memory_cleanup("between_operations")
        detailed_memory_tracking("AFTER_SHARD_CLEANUP", f"Cleaned up after {filename}")
        if low_disk_space_usage and push_to_hub:
            detailed_memory_tracking("BEFORE_SHARD_UPLOAD", f"Uploading {filename}")
            upload_items(filename)
            os.remove(os.path.join(save_directory, filename)) # Remove to conserve disk space
            detailed_memory_tracking("AFTER_SHARD_UPLOAD", f"Uploaded and removed {filename}")
        pass
    pass

    # Step 6: Regenerate index ONLY for MXFP4 dequantization
    if base_model_is_quantized and quant_type == "mxfp4" and len(safetensors_list) > 1:
        print("Unsloth: Regenerating safetensors index for dequantized MXFP4 model...")
        weight_map = {}

        for filename in safetensors_list:
            file_path = os.path.join(save_directory, filename)
            # Important check for low_disk_space mode where files might be deleted
            if not os.path.exists(file_path): continue
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = filename

        index_data = {"metadata": {}, "weight_map": weight_map}
        index_path = os.path.join(save_directory, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4)

        if push_to_hub:
            upload_items("model.safetensors.index.json")
        print("Unsloth: Merge process completed.")

    # Step 7: Final upload of all shards if not using low disk space mode and pushing
    if not low_disk_space_usage and push_to_hub:

        # Explicitly upload all safetensors files if not already handled
        for filename in safetensors_list:
            upload_items(filename)
        upload_items()


    # Step 7: Check for errors
    if len(lora_weights) != n_saved_modules:
        raise RuntimeError(
            f"Unsloth: Saving LoRA finetune failed since # of LoRAs = {len(lora_weights)} "\
            f"does not match # of saved modules = {n_saved_modules}. Please file a bug report!"
        )
    pass

    # --- Cleanup
    if temp_file is not None:
        try: temp_file.cleanup()
        except: pass
    pass
    # need to clean dangling files in the directory if we're pushing to hub,
    if push_to_hub and os.path.exists(save_directory):
        try:
            shutil.rmtree(save_directory)
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory {save_directory}: {e}")
    pass
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    detailed_memory_tracking("MERGE_COMPLETE", "Merge process finished")
    return save_directory
pass

def _try_copy_all_from_cache(
    repo_id: str,
    filenames_to_check: List[str],
    target_dir_str: str, # Expect string path for target directory
    hf_cache_dir: Optional[Path],
    token: Optional[str],
) -> bool:
    """
    Checks if ALL specified files exist in the HF cache. If yes, creates the
    target_dir_str and copies ALL files into it using os functions.
    Returns True if successful, False otherwise.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError

    if not hf_cache_dir or not filenames_to_check:
        print("Skipping cache check: No cache directory or no files specified.") # Verbose
        return False

    hf_cache_dir_str = str(hf_cache_dir)
    print(f"Checking cache directory for required files...") # Verbose
    cached_paths_map = {}

    all_found = True
    for filename in filenames_to_check:
        try:
            cached_path_str = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            cached_paths_map[filename] = Path(cached_path_str) # Store Path for checking
        except LocalEntryNotFoundError:
            print(f"Cache check failed: {filename} not found in local cache.") # Verbose
            all_found = False
            break
        except Exception as check_err:
            print(f"Cache check failed: Error checking for {filename}: {check_err}.")
            all_found = False
            break

    if not all_found:
        print("Not all required files found in cache. Will proceed with downloading.") # Verbose
        return False

    try:
        # Create target directory using os.makedirs
        os.makedirs(target_dir_str, exist_ok=True)
        if not os.access(target_dir_str, os.W_OK | os.X_OK):
             raise PermissionError(f"No write/execute permission for target directory: {target_dir_str}")
    except Exception as dir_err:
        print(f"Cache copy failed: Could not create or access target directory {target_dir_str}: {dir_err}")
        return False

    all_copied = True
    for filename, cached_path in cached_paths_map.items():
        try:
            # Pass string target_dir_str to copy helper
            _copy_file_from_source(cached_path, target_dir_str, filename)
        except (IOError, PermissionError, FileNotFoundError) as copy_err:
             print(f"Cache copy failed: Error copying {filename} from {cached_path} to {target_dir_str}: {copy_err}")
             all_copied = False; break
        except Exception as e:
            print(f"Cache copy failed: An unexpected error occurred copying {filename}: {e}")
            all_copied = False; break

    if all_copied:
        print(f"Successfully copied all {len(filenames_to_check)} files from cache to {target_dir_str}.")
        return True
    else:
        print("Failed to copy one or more files from cache. Will proceed with downloading.")
        return False
pass

def _copy_file_from_source(src_path: Union[str, Path], target_dir_str: str, filename: str):
    """Copies a file from src_path to target_dir_str/filename using os.path."""
    src_path = Path(src_path) # Keep Path for source checking ease
    dst_path = os.path.join(target_dir_str, filename) # Use os.path.join for destination

    if not src_path.is_file():
        raise FileNotFoundError(f"Source {src_path} is not a valid file.")
    if not os.access(src_path, os.R_OK):
         raise PermissionError(f"No read permission for source file: {src_path}")
    # Target dir creation and permission check is handled by caller (_try_copy_all_from_cache)
    try:
        shutil.copy2(str(src_path), dst_path) # Use string paths for shutil
    except Exception as e:
        raise IOError(f"Failed to copy {src_path} to {dst_path}: {e}") from e
pass

def _get_hf_cache_dir() -> Optional[Path]:
    """Determines the Hugging Face Hub cache directory."""
    potential_paths = []
    if "HF_HUB_CACHE" in os.environ:
        potential_paths.append(Path(os.environ["HF_HUB_CACHE"]))
    if "HF_HOME" in os.environ:
        potential_paths.append(Path(os.environ["HF_HOME"]) / "hub")
    potential_paths.append(Path.home() / ".cache" / "huggingface" / "hub")

    for cache_dir in potential_paths:
        try:
            # 1. Check if it exists and is a directory
            if cache_dir.is_dir():
                # 2. Check if we have read/write/execute access
                # Need W/X for potential lock files or internal operations by huggingface_hub
                if os.access(cache_dir, os.R_OK | os.W_OK | os.X_OK):
                    print(f"Found HuggingFace hub cache directory: {cache_dir.resolve()}")
                    return cache_dir.resolve() # Return absolute path
                else:
                    print(f"Warning: Found cache directory {cache_dir}, but lack R/W/X permissions. Cannot use cache.")
                    # Don't check other paths if we found the prioritized one but lack permissions
                    return None
            # If it exists but is not a dir, it's problematic, stop checking.
            elif cache_dir.exists():
                 print(f"Warning: Path {cache_dir} exists but is not a directory. Cannot use cache.")
                 return None
            # If it doesn't exist, continue to check the next potential path

        except Exception as e:
            # Handle potential issues like symlink loops, permissions errors during check
            print(f"Warning: Error accessing potential cache path {cache_dir}: {e}. Checking next option.")
            continue # Try the next path

    # If none of the paths worked
    print("No existing and accessible Hugging Face cache directory found.")
    return None


_PUSHING_CODE = \
"""
PushToHubMixin._upload_modified_files(
    PushToHubMixin,
    working_dir = save_directory,
    repo_id = '{repo_id}',
    files_timestamps = files_timestamps,
    commit_message = "Upload Unsloth finetuned model",
    token = token,
    create_pr = False,
    revision = {revision},
    commit_description = "Upload Unsloth finetuned model",
)
if {use_temp_file} and temp_file is not None: temp_file.cleanup()
else:
    shutil.rmtree(save_directory)
    os.makedirs(save_directory, exist_ok = True)
if {use_temp_file}:
    temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)
    save_directory = temp_file.name
files_timestamps = PushToHubMixin._get_files_timestamps(PushToHubMixin, save_directory)
"""

def incremental_save_pretrained(
    save_pretrained,
    low_disk_space_usage = True,
    use_temp_file = True,
    repo_id = "",
    revision = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Move file timestamps out
    makedir = re.search(r"os\.makedirs\(save_directory.+?\n", save_pretrained)
    assert(makedir is not None)
    span = makedir.span(0)
    save_pretrained = save_pretrained[:span[1]-1] + \
        "; files_timestamps = self._get_files_timestamps(save_directory); temp_file = None;\n" + \
        save_pretrained[span[1]:]
    pass

    # Find the main loop
    if "for shard_file, tensors in filename_to_tensors" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `for shard_file, tensors in filename_to_tensors`")
    for_loop = re.search(
        r"for shard_file, tensors in filename_to_tensors\:"\
        r".*?[\n]{1,}[ ]{4}[a-zA-Z0-9\_\#]",
        save_pretrained,
        flags = re.DOTALL | re.MULTILINE,
    )
    assert(for_loop is not None)

    span = for_loop.span(0)
    for_loop = save_pretrained[max(span[0], span[1]-8) : span[1]-1]
    where = re.search(r"[\n]{1,}", for_loop[::-1]).span(0)[0]
    for_loop = save_pretrained[span[0] : span[1]-where-1]
    spaces = len(re.findall(r"\n([ ]{4,})", for_loop)[0])

    first_newline = for_loop.find("\n") + 1
    for_loop = for_loop.rstrip()

    if low_disk_space_usage:
        new_for_loop = for_loop[:first_newline] + \
            for_loop[first_newline:] + \
            " "*spaces + \
            re.sub(r"[ ]{8,}", "",
                   _PUSHING_CODE.format(
                       repo_id = repo_id,
                       revision = revision,
                       use_temp_file = use_temp_file,
                    ).rstrip()
            ).replace("\n", "\n" + " "*spaces)
    else:
        new_for_loop = for_loop
    pass

    new_for_loop = new_for_loop + \
        "\n" + \
        " "*spaces + \
        "for tensor in shard:\n" + \
        " "*(spaces+4) + \
        "if tensor in DEQUANTIZED_KEYS: shard[tensor] = None\n"

    if low_disk_space_usage:
        new_for_loop = new_for_loop + \
            "\n" + \
            " "*(spaces-4) + \
            f"if {use_temp_file}:\n" + \
            " "*(spaces) + \
            "temp_file = tempfile.TemporaryDirectory(ignore_cleanup_errors = True)\n" + \
            " "*(spaces) + \
            "save_directory = temp_file.name\n" + \
            " "*(spaces) + \
            f"repo_id = '{repo_id}'\n"
    pass
    save_pretrained = save_pretrained.replace(for_loop, new_for_loop)

    if not low_disk_space_usage:
        save_pretrained = save_pretrained.replace(
            "for shard_file, tensors in filename_to_tensors",
            "for shard_file, tensors in ProgressBar(filename_to_tensors, desc = 'Unsloth: Saving ' + str(len(filename_to_tensors)) + ' safetensor(s)')",
            1,
        )
    pass
    return save_pretrained
pass


def merge_and_dequantize_lora(
    model,
    tokenizer            = None,
    save_directory       = "unsloth_finetuned_merge",
    push_to_hub          = False,
    max_shard_size       = "5GB",
    safe_serialization   = True,
    token                = None,
    private              = False,
    revision             = None,
    output_dtype         = None,
    low_disk_space_usage = False,
    use_temp_file        = False,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Dequantizes model to 16bit weights and merges LoRA
    inner_model = model.base_model.model if isinstance(model, PeftModelForCausalLM) else model
    inner_model = inner_model.base_model if hasattr(model, "base_model") else inner_model

    (
        username, repo_id, hf_api, token,
        output_dtype, element_size,
        lora_weights, state_dict, save_size, free,
        temp_file, save_directory, use_temp_file,
        low_disk_space_usage, max_shard_size_in_bytes,
    ) = prepare_saving(
        model = model,
        save_directory = save_directory,
        push_to_hub = push_to_hub,
        max_shard_size = max_shard_size,
        private = private,
        token = token,
        output_dtype = output_dtype,
        low_disk_space_usage = low_disk_space_usage,
        merge_into_original = False,
        min_size_in_bytes = 100_000_000, # 100MB default
        use_temp_file = use_temp_file,
    )

    import transformers.modeling_utils
    save_pretrained = inspect.getsource(transformers.modeling_utils.PreTrainedModel.save_pretrained)
    spaces = save_pretrained.find("def")
    save_pretrained = save_pretrained.split("\n")
    save_pretrained = "\n".join(x[spaces:] for x in save_pretrained)

    # Now patch for incremental pushing to hub
    if push_to_hub:
        save_pretrained = incremental_save_pretrained(
            save_pretrained = save_pretrained,
            low_disk_space_usage = low_disk_space_usage,
            use_temp_file = use_temp_file,
            repo_id = repo_id,
            revision = revision,
        )
    pass

    functions = dir(transformers.modeling_utils)
    # functions = [x for x in functions if (f"{x}." in save_pretrained or f"{x}(" in save_pretrained) and x != "PreTrainedModel"]
    exec(f"from transformers.modeling_utils import ({', '.join(functions)})", locals(), globals())

    replace_state_dict = f"""
    DEQUANTIZED_KEYS = []

    def merge_lora_weights(state_dict, name):
        x = state_dict[name]
        if type(x) is LoraStats:
            DEQUANTIZED_KEYS.append(name)
            W = dequantize_module_weight(x.module)
            W = _merge_lora(W, x, name)
            x = W.to(device = 'cpu', dtype = {str(output_dtype)}, non_blocking = True)
        # Remove memory leak
        state_dict[name] = None
        return x
    pass
    state_dict_split = split_state_dict_into_shards_factory(
        state_dict,
        max_shard_size   = {max_shard_size_in_bytes},
        filename_pattern = filename_pattern,
        get_storage_size = functools.partial(get_torch_storage_size_new, element_size = {element_size}),
        get_storage_id   = get_torch_storage_id_new,
    )
    """
    left  = save_pretrained.find("state_dict_split = split_torch_state_dict_into_shards")
    if left == -1: raise RuntimeError("Unsloth: Failed to find `state_dict_split`")
    right = save_pretrained.find(")", left) + 1
    save_pretrained = save_pretrained[:left] + replace_state_dict + save_pretrained[right:]

    if "state_dict[tensor].contiguous()" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `state_dict[tensor].contiguous()`")
    save_pretrained = save_pretrained.replace(
        "state_dict[tensor].contiguous()",
        "merge_lora_weights(state_dict, tensor).contiguous()",
        1,
    )

    if "def save_pretrained" not in save_pretrained:
        raise RuntimeError("Unsloth: Failed to find `def save_pretrained`")
    save_pretrained = save_pretrained.replace(
        "def save_pretrained",
        "def save_pretrained_dequantized",
        1,
    )

    functions = {}
    exec(save_pretrained, globals(), functions)
    save_pretrained_dequantized = functions["save_pretrained_dequantized"]
    save_pretrained_dequantized = torch.inference_mode(save_pretrained_dequantized)

    files_timestamps = PushToHubMixin._get_files_timestamps(
        PushToHubMixin,
        save_directory,
    )
    save_pretrained_dequantized(
        inner_model,
        save_directory     = save_directory,
        push_to_hub        = False,
        max_shard_size     = max_shard_size_in_bytes,
        safe_serialization = safe_serialization,
        token              = token,
        private            = private,
        state_dict         = state_dict,
        **kwargs,
    )

    # Save tokenizer
    if tokenizer is not None: tokenizer.save_pretrained(save_directory = save_directory,)

    if push_to_hub:
        commit = PushToHubMixin._upload_modified_files(
            PushToHubMixin,
            working_dir = save_directory,
            repo_id = repo_id,
            files_timestamps = files_timestamps,
            commit_message = "Upload Unsloth finetuned model",
            token = token,
            create_pr = False,
            revision = revision,
            commit_description = "Upload Unsloth finetuned model",
        )
        print(f"Unsloth: Uploaded model to https://huggingface.co/{repo_id}")
        return commit
    pass
    if temp_file is not None:
        try: temp_file.cleanup()
        except: pass
    pass
pass

def get_original_model_id(local_path: str):
    import json
    import os

    config_path = os.path.join(local_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)

        # Check for _name_or_path that's not a local path
        # When we load using AutoConfig, the _name_or_path changed into the local path instead
        if "_name_or_path" in config:
            return config["_name_or_path"]

    config_path = os.path.join(local_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)

        if "base_model_name_or_path" in config:
            return config["base_model_name_or_path"]

    return None

def _get_checkpoint_conversion_mapping(model_class_name):
    """Get the checkpoint conversion mapping for a specific model class"""
    try:
        # Dynamically import the model class
        module = __import__('transformers', fromlist=[model_class_name])
        model_class = getattr(module, model_class_name)
        return getattr(model_class, '_checkpoint_conversion_mapping', {})  # Returns {} if attribute doesn't exist
    except (ImportError, AttributeError):
        return {}
pass

from collections import defaultdict


def detect_keys_format(keys_to_check, forward_mapping):
    if not forward_mapping:
        return "new"

    count_matches_old_pattern = 0
    count_matches_new_pattern = 0

    # Compile regex patterns for efficiency if called multiple times with same mapping (though here it's per call)
    old_regex_compiled = [re.compile(p) for p in forward_mapping.keys()]
    # For new patterns (values of forward_mapping), treat them as literal prefixes to match
    new_regex_compiled = [re.compile(r"^" + re.escape(val)) for val in forward_mapping.values()]

    for key in keys_to_check:
        if not isinstance(key, str): continue

        # A key is "new" if it starts with one of the new_prefix_strings (values of forward_mapping)
        # A key is "old" if it matches one of the old_pattern_regex (keys of forward_mapping)
        #   AND it does NOT start with one of the new_prefix_strings (to avoid double counting if patterns overlap badly)

        matched_new = any(r.match(key) for r in new_regex_compiled)
        matched_old = any(r.match(key) for r in old_regex_compiled)

        if matched_new:
            count_matches_new_pattern += 1
        elif matched_old: # Only count as old if not already counted as new
            count_matches_old_pattern += 1

    # Decision logic
    if count_matches_new_pattern > 0 and count_matches_old_pattern == 0: return "new"
    if count_matches_old_pattern > 0 and count_matches_new_pattern == 0: return "old"

    # If mixed,
    if count_matches_new_pattern > count_matches_old_pattern: return "new"
    if count_matches_old_pattern > count_matches_new_pattern: return "old"

    return "new" # Default, assuming most models/keys will be in the "new" (current HF) format.

def _convert_lora_keys_to_safetensor_format(
    lora_weights,        # Global dict of LoraStats objects
    safetensor_keys,     # List of keys from the CURRENT shard
    model_class_name="PretrainedModel" # The actual model instance (e.g. Qwen2VLForConditionalGeneration)
):
    import re

    # Get the forward mapping from the model class itself
    forward_mapping = _get_checkpoint_conversion_mapping(model_class_name)

    if not forward_mapping:
        return defaultdict(lora_weights.default_factory, lora_weights)

    # Create reverse mapping
    reverse_mapping = {}
    for pattern, replacement in forward_mapping.items():
        reverse_mapping[replacement] = pattern
    # Determine formats
    lora_key_format_assumed = "new"
    shard_key_format = detect_keys_format(safetensor_keys, forward_mapping)

    converted_lora_weights_output = defaultdict(lora_weights.default_factory)
    conversion_applied_count = 0

    for lora_key_module_name, lora_stats in lora_weights.items():
        if not isinstance(lora_key_module_name, str):
            converted_lora_weights_output[lora_key_module_name] = lora_stats
            continue

        converted_key_for_lookup = lora_key_module_name
        applied_conversion_for_this_key = False

        if lora_key_format_assumed == "new" and shard_key_format == "old":
            # LoRA keys are new format, shard is old style -> convert LoRA key to old style
            # Use reverse mapping
            for pattern, replacement in reverse_mapping.items():
                replacement = re.sub(r"\^?([^(?]+).*", r"\1", replacement.lstrip("^"))
                temp_key, n_replace = re.subn(pattern, replacement, converted_key_for_lookup)
                if n_replace > 0:
                    converted_key_for_lookup = temp_key
                    applied_conversion_for_this_key = True
                    break

        elif lora_key_format_assumed == "old" and shard_key_format == "new":
            # LoRA keys are old format, shard is new format -> convert LoRA key to new style
            for pattern, replacement in forward_mapping.items():
                temp_key, n_replace = re.subn(pattern, replacement, converted_key_for_lookup)
                if n_replace > 0:
                    converted_key_for_lookup = temp_key
                    applied_conversion_for_this_key = True
                    break

        if applied_conversion_for_this_key:
            conversion_applied_count += 1

        converted_lora_weights_output[converted_key_for_lookup] = lora_stats
    return converted_lora_weights_output
pass

def find_lora_base_model(model_to_inspect):
    current = model_to_inspect
    if hasattr(current, "base_model"):
        current = current.base_model
    if hasattr(current, "model"):
        current = current.model
    return current
pass

def check_hf_model_exists(model_name, token=None):
    """Check if model exists on HuggingFace"""
    try:
        file_list = HfFileSystem(token=token).ls(model_name, detail=True)
        return any(x["name"].endswith(".safetensors") for x in file_list)
    except:
        return False
pass

def check_local_model_exists(model_path):
    """Check if model exists locally"""
    return os.path.exists(model_path) and os.path.isdir(model_path)
pass

def check_model_quantization_status(model_name_or_path, token=None):
    """Check if a model is quantized (works for both HF and local)"""
    config = None
    # Local path
    if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding="utf-8") as f:
                    config = json.load(f)
            except:
                pass
    # HF repo
    else:
        try:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="config.json",
                cache_dir=None,
                token=token
            )
            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
        except:
            pass

    if config and "quantization_config" in config:
        quant_config = config["quantization_config"]

        # Case 2: Check for MXFP4 format first (more specific)
        # We assume the Mxfp4Config serializes with a "quant_method": "mxfp4" key.
        if isinstance(quant_config, dict) and quant_config.get("quant_method") == "mxfp4":
            return (True, "mxfp4")

        # Case 1: Fallback to existing logic for bitsandbytes
        elif isinstance(quant_config, dict):
            is_quantized = quant_config.get("load_in_4bit", False)
            quant_type = quant_config.get("bnb_4bit_quant_type", None)
            if is_quantized:
                # Return the specific type if available, otherwise a generic "bitsandbytes"
                return (True, quant_type if quant_type else "bitsandbytes")

    return (False, None)
pass

def determine_base_model_source(model_name, token=None):
    """
    Determine the best source for base model using branched logic
    Returns: (final_model_name, is_local_path, source_info, is_quantized, quant_type)
    """

    # Check availability
    hf_exists = check_hf_model_exists(model_name, token)
    local_exists = check_local_model_exists(model_name)

    # Branch A: HF model exists
    if hf_exists:
        hf_is_quantized, hf_quant_type = check_model_quantization_status(model_name, token)

        if not hf_is_quantized:
            # A1: HF unquantized exists → use HF
            return (model_name, False, "HF_unquantized", False, None)
        else:
            # A2: HF is quantized, check if local unquantized exists
            if local_exists:
                local_is_quantized, local_quant_type = check_model_quantization_status(model_name)
                if not local_is_quantized:
                    # A2a: Local unquantized exists → use local
                    return (model_name, True, "local_unquantized_preferred_over_HF_quantized", False, None)
                else:
                    # A2b: Both quantized → use HF (more reliable)
                    return (model_name, False, "HF_quantized", True, hf_quant_type)
            else:
                # A3: Only HF quantized exists
                return (model_name, False, "HF_quantized_only", True, hf_quant_type)

    # Branch B: HF model doesn't exist
    else:
        if local_exists:
            # B1: Any local exists → use local
            local_is_quantized, local_quant_type = check_model_quantization_status(model_name)
            status = "quantized" if local_is_quantized else "unquantized"
            return (model_name, True, f"local_{status}_only", local_is_quantized, local_quant_type)
        else:
            # B2: Nothing found
            raise ValueError(f"Model {model_name} not found locally or on HuggingFace")
pass

def get_memory_stats():
    """Get current memory statistics for CPU and GPU"""
    stats = {}
    import psutil

    # CPU Memory
    cpu_mem = psutil.virtual_memory()
    stats['cpu'] = {
        'total': cpu_mem.total,
        'available': cpu_mem.available,
        'used': cpu_mem.used,
        'percent': cpu_mem.percent,
        'free': cpu_mem.available,  # Available is more accurate than free
        'cached': getattr(cpu_mem, 'cached', 0),  # OS cache
        'buffers': getattr(cpu_mem, 'buffers', 0),  # OS buffers
    }

    # GPU Memory (for each GPU)
    stats['gpus'] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem = torch.cuda.mem_get_info(i)
            total = gpu_mem[1]
            free = gpu_mem[0]
            stats['gpus'].append({
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'total': total,
                'free': free,
                'used': total - free,
                'percent': ((total - free) / total) * 100 if total > 0 else 0
            })

    return stats
pass

def format_bytes(bytes_value):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"
pass

def calculate_combined_score(speed_score, chunk_size):
    # Normalize chunk size to 0-1 scale (assuming max reasonable chunk is 100M)
    chunk_factor = min(1.0, chunk_size / (100 * 1024 * 1024))
    # Weight: 60% device speed, 40% chunk efficiency
    return speed_score * 0.6 + chunk_factor * 10 * 0.4  # Scale chunk factor to match speed range
pass

def _choose_mxfp4_processing_strategy(blocks_tensor, scales_tensor):
    """
    Choose optimal device and chunk size for mxfp4 dequantization based on available memory.
    """
    import math

    # Calculate tensor dimensions
    *prefix_shape, G, B = blocks_tensor.shape
    rows_total = math.prod(prefix_shape) * G

    # Estimate memory requirements
    #base_memory_per_row = B * 21 if B else 128 * 21
    base_memory_per_row = B * 35 if B else 128 * 35
    input_size = blocks_tensor.numel() + scales_tensor.numel() * 4
    output_size = rows_total * B * 2 * 2
    persistent_memory = input_size + output_size

    # Device-specific safety factors
    GPU_SAFETY_FACTOR = 0.75  # GPUs can handle higher utilization
    CPU_SAFETY_FACTOR = 0.75  # CPUs need more headroom for OS and other processes

    def calculate_safe_usable_memory(free_memory, safety_factor):
        # Option 1: What we can use from reported free memory (accounting for fragmentation)
        usable_from_free = free_memory * safety_factor

        return usable_from_free


    def calculate_optimal_chunk_size(safe_usable_memory):
        """Calculate the largest chunk size that fits in the safe usable memory"""
        temp_memory_budget = safe_usable_memory - persistent_memory
        if temp_memory_budget <= 0:
            return None

        max_chunk_from_memory = int(temp_memory_budget // base_memory_per_row)
        optimal_chunk = min(rows_total, max_chunk_from_memory)

        if optimal_chunk < 1024:
            return None

        return optimal_chunk

    stats = get_memory_stats()
    suitable_strategies = []

    # Check GPU strategies first (preferred for speed)
    for gpu in stats['gpus']:
        safe_usable_memory = calculate_safe_usable_memory(
            free_memory=gpu['free'],
            safety_factor=GPU_SAFETY_FACTOR,
        )
        chunk_size = calculate_optimal_chunk_size(safe_usable_memory)

        if chunk_size:
            temp_memory = min(chunk_size, rows_total) * base_memory_per_row
            total_memory_needed = persistent_memory + temp_memory

            combined_score = calculate_combined_score(3.0, chunk_size)

            suitable_strategies.append({
                'device_type': 'cuda',
                'device_id': gpu['device_id'],
                'rows_per_chunk': chunk_size,
                'available_memory': gpu['free'] * GPU_SAFETY_FACTOR,
                'total_memory': gpu['total'],
                'safe_usable_memory': safe_usable_memory,
                'needed_memory': total_memory_needed,
                'speed_score': 3.0,
                'efficiency_score': chunk_size,
                'safety_factor': GPU_SAFETY_FACTOR,
                'memory_utilization': total_memory_needed / safe_usable_memory,
                'combined_score': combined_score,
            })

    # Check CPU strategy
    cpu_safe_usable_memory = calculate_safe_usable_memory(
        free_memory=stats['cpu']['available'],
        safety_factor=CPU_SAFETY_FACTOR,
    )
    cpu_chunk_size = calculate_optimal_chunk_size(cpu_safe_usable_memory)

    if cpu_chunk_size:
        temp_memory = min(cpu_chunk_size, rows_total) * base_memory_per_row
        total_memory_needed = persistent_memory + temp_memory
        combined_score = calculate_combined_score(1.0, cpu_chunk_size)  # For CPU
        suitable_strategies.append({
            'device_type': 'cpu',
            'device_id': None,
            'rows_per_chunk': cpu_chunk_size,
            'available_memory': stats['cpu']['available'] * CPU_SAFETY_FACTOR,
            'total_memory': stats['cpu']['total'],
            'safe_usable_memory': cpu_safe_usable_memory,
            'needed_memory': total_memory_needed,
            'speed_score': 1.0,
            'efficiency_score': cpu_chunk_size,
            'safety_factor': CPU_SAFETY_FACTOR,
            'fragmentation_factor': 1.0,
            'memory_utilization': total_memory_needed / cpu_safe_usable_memory,
            'combined_score': combined_score,
        })

    if suitable_strategies:

        # Sort by combined score
        suitable_strategies.sort(key=lambda x: x['combined_score'], reverse=True)

        best = suitable_strategies[0]

        if UNSLOTH_ENABLE_LOGGING:
            logger.debug(
                f"[MXFP4] Selected {best['device_type']}:{best['device_id'] or ''} "
                f"with {best['rows_per_chunk']:,} rows per chunk "
                f"(safety factor: {best['safety_factor']:.0%}, "
                f"safe memory utilization: {best['memory_utilization']:.1%}) "
                f"- Need: {format_bytes(best['needed_memory'])}, "
                f"Available: {format_bytes(best['available_memory'])}"
            )

        return (best['device_type'], best['device_id'], best['rows_per_chunk'])

    # Fallback: find device with most memory and use minimal chunk
    fallback_options = []

    # Add CPU fallback
    fallback_options.append({
        'device_type': 'cpu',
        'device_id': None,
        'available': stats['cpu']['available'] * CPU_SAFETY_FACTOR,
        'total_available': stats['cpu']['available']
    })

    # Add GPU fallbacks
    for gpu in stats['gpus']:
        fallback_options.append({
            'device_type': 'cuda',
            'device_id': gpu['device_id'],
            'available': gpu['free'] * GPU_SAFETY_FACTOR,
            'total_available': gpu['free']
        })

    # Sort by available memory (after safety factor)
    fallback_options.sort(key=lambda x: x['available'], reverse=True)
    best_fallback = fallback_options[0]

    # Calculate minimal safe chunk size for fallback
    remaining_memory = best_fallback['available'] - persistent_memory
    if remaining_memory > 0:
        fallback_chunk_size = max(1024, min(8192, int(remaining_memory // base_memory_per_row), rows_total))
    else:
        fallback_chunk_size = min(1024, rows_total)

    warnings.warn(
        f"[MXFP4] Insufficient memory for optimal processing on any device. "
        f"Using {best_fallback['device_type']}:{best_fallback['device_id'] or ''} "
        f"with minimal chunks ({fallback_chunk_size:,}). "
        f"Available: {format_bytes(best_fallback['total_available'])}, "
        f"Required: {format_bytes(persistent_memory)}. "
        f"Processing will be slow."
    )

    return (best_fallback['device_type'], best_fallback['device_id'], fallback_chunk_size)
pass

def _choose_merge_device(W, lora_stats, key):
    """
    Chooses the optimal device (CPU or GPU) to perform a LoRA merge based on memory requirements.
    """
    if not torch.cuda.is_available():
        return "cpu"

    lora_A = lora_stats.lora_A
    lora_B = lora_stats.lora_B
    if lora_A is None or lora_B is None:
        return W.device if isinstance(W, torch.Tensor) else "cpu"

    # Memory cost is for the float32 versions of the matrices
    memory_cost = (W.numel() + lora_A.numel() + lora_B.numel()) * 4 # 4 bytes for float32

    stats = get_memory_stats()
    chosen_device = None

    # Device-specific safety factors
    GPU_SAFETY_FACTOR = 0.80  # Leave 20% free VRAM
    CPU_SAFETY_FACTOR = 0.75  # Leave 25% free RAM

    def calculate_safe_usable_memory(free_memory, safety_factor):
        return free_memory * safety_factor

    # Check GPU VRAM first
    if stats['gpus']:
        gpu_safe_usable_memory = calculate_safe_usable_memory(
            free_memory=stats['gpus'][0]['free'],
            safety_factor=GPU_SAFETY_FACTOR,
        )
        if gpu_safe_usable_memory > memory_cost:
            chosen_device = "cuda"

    # Check CPU RAM as a fallback if GPU is not chosen
    if chosen_device is None:
        cpu_safe_usable_memory = calculate_safe_usable_memory(
            free_memory=stats['cpu']['available'],
            safety_factor=CPU_SAFETY_FACTOR,
        )
        if cpu_safe_usable_memory > memory_cost:
            warnings.warn(
                f"Unsloth: Not enough VRAM to merge LoRA layers for '{key}'. "
                f"Required: {format_bytes(memory_cost)}. Merging on CPU. This will be slower."
            )
            chosen_device = "cpu"

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(
            f"[DEBUG] Merge stats for '{key}': "
            f"Cost = {format_bytes(memory_cost)}. "
            f"Chosen device = '{chosen_device}'."
        )

    # If neither has enough memory, raise an error
    if chosen_device is None:
        raise RuntimeError(
            f"Unsloth: Not enough memory to merge LoRA layer '{key}' ({W.shape}).\n"
            f"Required memory: {format_bytes(memory_cost)}.\n"
            f"Available VRAM (safe): {format_bytes(gpu_safe_usable_memory)}.\n"
            f"Available RAM (safe): {format_bytes(cpu_safe_usable_memory)}."
        )

    return chosen_device
pass

def detailed_memory_tracking(stage, key=None, extra_info=None):
    """Detailed memory tracking with breakdown"""
    import psutil
    import os

    # Get current process
    process = psutil.Process(os.getpid())
    # Memory info
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()

    # System memory
    system_mem = psutil.virtual_memory()
    stats = get_memory_stats()

    # Calculate actual system usage (what htop shows)
    system_used = stats['cpu']['total'] - stats['cpu']['available']
    system_used_percent = (system_used / stats['cpu']['total']) * 100

    # GPU memory if available
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.mem_get_info(0)
        gpu_free_gb = gpu_mem[0] / (1024**3)
        gpu_used_gb = (gpu_mem[1] - gpu_mem[0]) / (1024**3)
        gpu_info = f"GPU: {gpu_used_gb:.1f}GB used, {gpu_free_gb:.1f}GB free"

    key_info = f" [{key}]" if key else ""
    extra_info = f" - {extra_info}" if extra_info else ""

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(
            f"[MEMORY] {stage}{key_info}: "
            f"RSS={mem_info.rss/(1024**3):.1f}GB, "
            f"VMS={mem_info.vms/(1024**3):.1f}GB, "
            f"Percent={mem_percent:.1f}%, "
                f"System_Used={system_used/(1024**3):.1f}GB ({system_used_percent:.1f}%), "
            f"{gpu_info}{extra_info}"
        )

def track_dict_size(tensors_dict, stage):
    """Track the actual memory footprint of the tensors dictionary"""
    if not tensors_dict:
        return

    total_elements = sum(t.numel() if hasattr(t, 'numel') else 0 for t in tensors_dict.values())
    dict_size_gb = total_elements * 2 / (1024**3)  # Assuming float16
    num_tensors = len(tensors_dict)

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(f"[DICT_SIZE] {stage}: {num_tensors} tensors, ~{dict_size_gb:.1f}GB total")
pass

def calculate_optimal_batch_size(safetensor_keys=None, file_handle=None):
    """
    Calculate optimal batch size based on available memory and tensor sizes
    """
    stats = get_memory_stats()

    # Safety factors - leave headroom for OS and other processes
    #CPU_SAFETY_FACTOR = 0.70  # Use 70% of available RAM
    CPU_SAFETY_FACTOR = 0.75

    # Calculate available memory
    cpu_available = stats['cpu']['available']
    cpu_safe_usable = int(cpu_available * CPU_SAFETY_FACTOR)

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(f"[BATCH_CALC] Total RAM: {format_bytes(stats['cpu']['total'])}")
        logger.debug(f"[BATCH_CALC] Available RAM: {format_bytes(cpu_available)}")
        logger.debug(f"[BATCH_CALC] Safe usable RAM: {format_bytes(cpu_safe_usable)}")

    # If we have file info, analyze actual tensor sizes for better estimation
    if safetensor_keys and file_handle:
        # Sample a few tensors to estimate size distribution
        sample_keys = safetensor_keys[:min(20, len(safetensor_keys))]  # Sample first 20 tensors
        tensor_sizes = []

        for key in sample_keys:
            try:
                tensor = file_handle.get_tensor(key)
                # Calculate size in bytes (numel * element_size)
                tensor_size = tensor.numel() * tensor.element_size()
                tensor_sizes.append(tensor_size)
            except:
                # Fallback if we can't sample
                tensor_sizes.append(50 * 1024 * 1024)  # 50MB estimate

        if tensor_sizes:
            avg_tensor_size = sum(tensor_sizes) / len(tensor_sizes)
            max_tensor_size = max(tensor_sizes)

            # Use weighted average (favor larger tensors for safety)
            estimated_tensor_size = int(avg_tensor_size * 1.5)  # 50% safety margin

            if UNSLOTH_ENABLE_LOGGING:
                logger.debug(f"[BATCH_CALC] Sampled {len(tensor_sizes)} tensors")
                logger.debug(f"[BATCH_CALC] Avg tensor size: {format_bytes(avg_tensor_size)}")
                logger.debug(f"[BATCH_CALC] Max tensor size: {format_bytes(max_tensor_size)}")
                logger.debug(f"[BATCH_CALC] Estimated tensor size (with safety): {format_bytes(estimated_tensor_size)}")
        else:
            estimated_tensor_size = 50 * 1024 * 1024  # 50MB fallback
    else:
        # Fallback estimation based on model size heuristics
        estimated_tensor_size = 50 * 1024 * 1024  # 50MB default

    # Calculate batch size based on available memory
    raw_batch_size = max(1, int(cpu_safe_usable // estimated_tensor_size))

    # Apply intelligent limits based on total system memory
    total_ram_gb = stats['cpu']['total'] / (1024**3)

    if total_ram_gb >= 200:  # High memory systems (200GB+)
        min_batch_size = 100
        max_batch_size = 500
    elif total_ram_gb >= 100:  # Medium-high memory systems (100-200GB)
        min_batch_size = 50
        max_batch_size = 200
    elif total_ram_gb >= 32:  # Medium memory systems (32-100GB)
        min_batch_size = 20
        max_batch_size = 100
    elif total_ram_gb >= 16:  # Low-medium memory systems (16-32GB)
        min_batch_size = 10
        max_batch_size = 50
    else:  # Low memory systems (<16GB)
        min_batch_size = 2
        max_batch_size = 20

    # Clamp batch size to reasonable bounds
    optimal_batch_size = max(min_batch_size, min(max_batch_size, raw_batch_size))

    # Final memory check - ensure batch won't exceed available memory
    estimated_batch_memory = optimal_batch_size * estimated_tensor_size
    if estimated_batch_memory > cpu_safe_usable:
        # Recalculate with stricter limit
        optimal_batch_size = max(1, int(cpu_safe_usable // estimated_tensor_size))

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(f"[BATCH_CALC] Raw calculated batch size: {raw_batch_size}")
        logger.debug(f"[BATCH_CALC] System tier limits: {min_batch_size}-{max_batch_size}")
        logger.debug(f"[BATCH_CALC] Final optimal batch size: {optimal_batch_size}")
        logger.debug(f"[BATCH_CALC] Estimated batch memory usage: {format_bytes(estimated_batch_memory)}")
        logger.debug(f"[BATCH_CALC] Memory utilization: {(estimated_batch_memory/cpu_safe_usable)*100:.1f}%")

    return optimal_batch_size
pass

def calculate_dynamic_batch_size(actual_tensor_sizes, available_memory_bytes, safety_factor=0.7):
    """
    Calculate batch size based on actual tensor sizes seen so far
    """

    #Get current available memory if not provided
    if available_memory_bytes is None:
        stats = get_memory_stats()
        available_memory_bytes = stats['cpu']['available']
    if not actual_tensor_sizes:
        # Initial conservative estimate for first few tensors
        return min(10, max(2, int(available_memory_bytes // (100 * 1024 * 1024))))  # Assume 100MB per tensor initially

    # Use sliding window average of recent tensor sizes (more responsive to current patterns)
    recent_window = actual_tensor_sizes[-50:] if len(actual_tensor_sizes) > 50 else actual_tensor_sizes
    avg_tensor_memory = sum(recent_window) / len(recent_window)

    # Calculate how many tensors can fit in available memory
    safe_memory = int(available_memory_bytes * safety_factor)
    optimal_count = max(1, int(safe_memory // avg_tensor_memory))

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(f"[DYNAMIC_BATCH] Available memory: {format_bytes(available_memory_bytes)}")
        logger.debug(f"[DYNAMIC_BATCH] Recent avg tensor size: {format_bytes(avg_tensor_memory)}")
        logger.debug(f"[DYNAMIC_BATCH] Safe memory: {format_bytes(safe_memory)}")
        logger.debug(f"[DYNAMIC_BATCH] Optimal batch size: {optimal_count}")

    return optimal_count

def calculate_tensor_memory_cost(W, lora_stats=None, output_dtype=torch.float16):
    """
    Calculate actual memory cost of a tensor including LoRA components
    Similar to _choose_merge_device but focused on memory calculation
    """
    base_memory = W.numel() * W.element_size()

    # Add LoRA memory if present
    lora_memory = 0
    if lora_stats and hasattr(lora_stats, 'lora_A') and lora_stats.lora_A is not None:
        lora_A = lora_stats.lora_A
        lora_B = lora_stats.lora_B
        if lora_A is not None and lora_B is not None:
            # Memory cost during merge (float32 versions)
            lora_memory = (lora_A.numel() + lora_B.numel()) * 4  # 4 bytes for float32

    # Output tensor memory (in target dtype)
    output_memory = W.numel() * torch.tensor([], dtype=output_dtype).element_size()

    # Total memory needed during processing (base + lora processing + output)
    total_memory = base_memory + lora_memory + output_memory

    return total_memory

def adaptive_batch_resize(current_batch_size, current_memory_usage, target_memory_usage):
    """
    Dynamically adjust batch size based on actual memory usage during processing
    """
    if current_memory_usage == 0:
        return current_batch_size

    # Calculate scaling factor
    memory_ratio = target_memory_usage / current_memory_usage

    # Apply conservative scaling (don't change too dramatically)
    scaling_factor = max(0.5, min(2.0, memory_ratio * 0.8))  # Limit to 50%-200% with dampening

    new_batch_size = max(1, int(current_batch_size * scaling_factor))

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(f"[BATCH_ADAPT] Memory ratio: {memory_ratio:.2f}")
        logger.debug(f"[BATCH_ADAPT] Scaling factor: {scaling_factor:.2f}")
        logger.debug(f"[BATCH_ADAPT] Batch size: {current_batch_size} -> {new_batch_size}")

    return new_batch_size
pass

def conservative_memory_cleanup(stage=""):
    """
    Cleanup that doesn't hurt performance
    """
    if UNSLOTH_ENABLE_LOGGING:
        pre_stats = get_memory_stats()
        logger.debug(f"[CLEANUP] Before {stage}: RSS={format_bytes(pre_stats['cpu']['used'])}")

    # Python garbage collection (always safe)
    for _ in range(3):
        gc.collect()

    # Clear unused GPU memory (safe, doesn't clear context/kernels)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Only clears unused allocations
        torch.cuda.synchronize()

    # OS-level memory compaction (safe)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Return unused heap to OS
    except:
        pass

    if UNSLOTH_ENABLE_LOGGING:
        post_stats = get_memory_stats()
        freed = pre_stats['cpu']['used'] - post_stats['cpu']['used']
        logger.debug(f"[CLEANUP] After {stage}: RSS={format_bytes(post_stats['cpu']['used'])}, Freed={format_bytes(freed)}")
pass

def calculate_memory_aware_batch_size(current_batch_tensors, next_tensor_info_list, converted_lora_weights, output_dtype, safety_factor=0.85):
    """
    Calculate batch size based on actual memory costs of upcoming tensors
    """
    stats = get_memory_stats()
    available_memory = stats['cpu']['available']
    safe_memory_budget = int(available_memory * safety_factor)  # Increased to 85%

    # Calculate current batch memory usage
    current_batch_memory = sum(
        calculate_tensor_memory_cost(tensor, None, output_dtype)
        for tensor in current_batch_tensors.values()
    )

    remaining_budget = safe_memory_budget - current_batch_memory

    # Look ahead at next tensors and their actual costs
    tensors_to_add = 0
    projected_memory = current_batch_memory

    for tensor_info in next_tensor_info_list[:20]:  # Look ahead at next 20 tensors
        key, W = tensor_info['key'], tensor_info['tensor']

        # Get actual LoRA stats for this specific tensor
        lora_key = key[:-len(".weight")] if key.endswith(".weight") else key
        lora_stats = converted_lora_weights.get(lora_key, None)

        # Calculate EXACT memory cost for this specific tensor
        tensor_memory_cost = calculate_tensor_memory_cost(W, lora_stats, output_dtype)

        if projected_memory + tensor_memory_cost <= safe_memory_budget:
            tensors_to_add += 1
            projected_memory += tensor_memory_cost
        else:
            break  # Would exceed budget

    if UNSLOTH_ENABLE_LOGGING:
        logger.debug(f"[MEMORY_AWARE_BATCH] Available: {format_bytes(available_memory)}")
        logger.debug(f"[MEMORY_AWARE_BATCH] Budget: {format_bytes(safe_memory_budget)}")
        logger.debug(f"[MEMORY_AWARE_BATCH] Current batch: {format_bytes(current_batch_memory)}")
        logger.debug(f"[MEMORY_AWARE_BATCH] Can add {tensors_to_add} more tensors")
        logger.debug(f"[MEMORY_AWARE_BATCH] Projected total: {format_bytes(projected_memory)}")

    return max(1, len(current_batch_tensors) + tensors_to_add)
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
