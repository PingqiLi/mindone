from dataclasses import is_dataclass

import pytest
import logging
from typing import Tuple, Union, Dict, List
import random
import numpy as np
import torch
from safetensors.torch import load_file
import mindspore as ms
from transformers import Wav2Vec2Processor, AutoProcessor
from transformers.models.hubert import HubertConfig, HubertForCTC
from transformers.modeling_outputs import ModelOutput as pt_ModelOutput
from datasets import load_dataset


def get_module_structure(model_path, target_modules):
    processor = AutoProcessor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation",
                           trust_remote_code=True)
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    # prepare inputs
    inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    pt_inputs = {key: torch.tensor(value) for key, value in inputs.items()}
    modules = {}
    def hook_fn(module, input, output):
        def extract_shapes(data):
            """递归提取数据的shape，如果是dict或dataclass则保持结构"""
            if isinstance(data, torch.Tensor):
                return tuple(data.shape)
            elif isinstance(data, dict):
                return {k: extract_shapes(v) for k, v in data.items()}
            elif is_dataclass(data):
                return {f.name: extract_shapes(getattr(data, f.name)) for f in fields(data)}
            elif isinstance(data, tuple):
                return tuple(extract_shapes(i) for i in data)
            else:
                return None

        module_name = module.__class__.__name__
        input_shape = [extract_shapes(i) for i in input]
        if len(input_shape) == 1:
            input_shape = input_shape[0]
        output_shape = extract_shapes(output)
        if module_name not in modules:
            modules[module_name] = {"input_shape": input_shape, "output_shape": output_shape}

    for name, module in model.named_modules():
        if module.__class__.__name__ in target_modules:
            module.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**pt_inputs)

    for module_name, inout_dict in modules.items():
        print(f"Module: {module_name}")
        print(f"Input Shape: {inout_dict['input_shape']}")
        print(f"Output Shape: {inout_dict['output_shape']}")
        print("-"*30)




if __name__ == "__main__":
    model_path = "/home/pqli/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots/ece5fabbf034c1073acae96d5401b25be96709d8"
    target_modules = [ 'HubertAttention', 'HubertAttnAdapterLayer', 'HubertEncoder',
                       'HubertEncoderLayer', 'HubertEncoderLayerStableLayerNorm',
                       'HubertEncoderStableLayerNorm', 'HubertFeatureEncoder',
                       'HubertFeatureExtractor', 'HubertFeatureProjection', 'HubertFeedForward',
                       'HubertGroupNormConvLayer', 'HubertLayerNormConvLayer',
                       'HubertNoLayerNormConvLayer', 'HubertPositionalConvEmbedding',
                       'HubertSamePadLayer']

    get_module_structure(model_path, target_modules)