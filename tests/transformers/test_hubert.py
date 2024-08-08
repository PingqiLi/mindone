import pytest
import logging
from typing import Tuple, Union, Dict, List

import numpy as np
import torch
from safetensors.torch import load_file
import mindspore as ms
from transformers import Wav2Vec2Processor
from transformers.models.hubert import HubertConfig, HubertForCTC as pt_HubertForCTC
from mindone.transformers.models.hubert import HubertForCTC as ms_HubertForCTC
from transformers.modeling_outputs import ModelOutput as pt_ModelOutput
from mindone.transformers.modeling_outputs import ModelOutput as ms_ModelOutput

logger = logging.getLogger(__name__)

THRESHOLD_FP16 = 1e-2
THRESHOLD_FP32 = 5e-3


@pytest.mark.parametrize(
    "name,mode,dtype",
    [
        ["facebook/hubert-base-ls960", ms.GRAPH_MODE, "fp32"],
        ["facebook/hubert-base-ls960", ms.GRAPH_MODE, "fp16"],
        ["facebook/hubert-base-ls960", ms.PYNATIVE_MODE, "fp32"],
        ["facebook/hubert-base-ls960", ms.PYNATIVE_MODE, "fp16"],
    ],
)
def test_hubert(model_name, mode, dtype):
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

    # init model
    config = HubertConfig.from_pretrained(model_name)
    pt_model = pt_HubertForCTC(config)
    ms_model = ms_HubertForCTC(config)

    state_dict = load_file(model_name)
    logger.info(">>> Loading pytorch model from %s", model_name)
    pt_model.load_state_dict(state_dict)
    logger.info(">>> Loading mindspore model from %s", model_name)
    ms_model.load_param_into_net(ms_model, state_dict)

    # convert model dtype
    _set_model_dtype(pt_model, ms_model, dtype)

    # get inputs
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    shape = (2, 16000)
    input_values = _generate_inputs(shape)
    inputs = processor(input_values, return_tensors="pt", sampling_rate=16000)

    with torch.no_grad():
        pt_outputs = pt_model(**inputs)
    ms_outputs = ms_model(**inputs)

    diffs = _compute_diffs(pt_outputs.hidden_states, ms_outputs.hidden_states)

    eps = THRESHOLD_FP16 if dtype == "fp16" else THRESHOLD_FP32
    assert (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"


_TORCH_FP16_BLACKLIST = (
    "LayerNorm",
    "Timesteps",
    "AvgPool2d",
    "Upsample2D",
    "ResnetBlock2D",
    "FirUpsample2D",
    "FirDownsample2D",
    "KDownsample2D",
    "AutoencoderTiny",
)


def _set_model_dtype(pt_modules_instance, ms_modules_instance, dtype):
    if dtype == "fp16":
        pt_modules_instance = pt_modules_instance.to(torch.float16)
        ms_modules_instance = _set_dtype(ms_modules_instance, ms.float16)
    elif dtype == "fp32":
        pt_modules_instance = pt_modules_instance.to(torch.float32)
        ms_modules_instance = _set_dtype(ms_modules_instance, ms.float32)
    else:
        raise NotImplementedError(f"Dtype {dtype} for model is not implemented")

    pt_modules_instance.eval()
    ms_modules_instance.set_train(False)

    if dtype == "fp32":
        return pt_modules_instance, ms_modules_instance

    # Some torch modules do not support fp16 in CPU, converted to fp32 instead.
    for _, submodule in pt_modules_instance.named_modules():
        if submodule.__class__.__name__ in _TORCH_FP16_BLACKLIST:
            logger.warning(
                f"Model '{pt_modules_instance.__class__.__name__}' has submodule {submodule.__class__.__name__} which doens't support fp16, converted to fp32 instead."
            )
            pt_modules_instance = pt_modules_instance.to(torch.float32)
            break

    return pt_modules_instance, ms_modules_instance


def _set_dtype(model, dtype):
    for p in model.get_parameters():
        p = p.set_dtype(dtype)
    return model


def _generate_inputs(shape: Tuple[int, ...]):
    return np.random.rand(*shape).astype(np.float32)


def _compute_diffs(pt_outputs: torch.Tensor, ms_outputs: ms.Tensor):
    if isinstance(pt_outputs, pt_ModelOutput):
        pt_outputs = tuple(pt_outputs.values())
    elif not isinstance(pt_outputs, (tuple, list)):
        pt_outputs = (pt_outputs,)
    if not isinstance(ms_outputs, (tuple, list)):
        ms_outputs = (ms_outputs,)

    diffs = []
    for p, m in zip(pt_outputs, ms_outputs):
        if isinstance(p, pt_ModelOutput):
            p = tuple(p.values())[0]

        p = p.detach().cpu().numpy()
        m = m.asnumpy()

        # relative error defined by Frobenius norm
        # dist(x, y) := ||x - y|| / ||y||, where ||·|| means Frobenius norm
        d = np.linalg.norm(p - m) / np.linalg.norm(p)

        diffs.append(d)

    return diffs

if __name__ == "__main__":
    pass