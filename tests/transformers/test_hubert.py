import numpy as np
import pytest
import torch
from typing import Tuple, Union, Dict, List
import mindspore as ms
from transformers.models.hubert import HubertConfig, HubertModel as pt_HubertModel
from mindone.transformers.models.hubert import HubertModel as ms_HubertModel
# from .modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

THRESHOLD_FP16 = 1e-2
THRESHOLD_FP32 = 5e-3


@pytest.mark.parametrize(
    "name,mode,dtype",
    [
        ["HubertModel_graph_fp32", 0, "fp32"],
        ["HubertModel_graph_fp16", 0, "fp16"],
        ["HubertModel_pynative_fp32", 1, "fp32"],
        ["HubertModel_pynative_fp16", 1, "fp16"],
    ],
)
def test_hubert(name, mode, dtype):
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

    # init model
    pt_model = pt_HubertModel.from_pretrained(name)
    ms_model = ms_HubertModel.from_pretrained(name)

    # get inputs
    inputs = _generate_inputs()


    with torch.no_grad():
        pt_outputs = pt_model(**pt_inputs_kwargs)
    ms_outputs = ms_model(**ms_inputs_kwargs)


    diffs = _compute_diffs(pt_outputs, ms_outputs)

    eps = THRESHOLD_FP16 if dtype == "fp16" else THRESHOLD_FP32
    assert (np.array(diffs) < eps).all(), f"Outputs({np.array(diffs).tolist()}) has diff bigger than {eps}"


def _set_model_dtype(model):
    pass


def _generate_inputs(shape=Union[int, Tuple[int, ...]], seed=42):
    pass


def _compute_diffs(pt_outputs: torch.Tensor, ms_outputs: ms.Tensor):
    if isinstance(pt_outputs, ModelOutput):
        pt_outputs = tuple(pt_outputs.values())
    elif not isinstance(pt_outputs, (tuple, list)):
        pt_outputs = (pt_outputs,)
    if not isinstance(ms_outputs, (tuple, list)):
        ms_outputs = (ms_outputs,)

    diffs = []
    for p, m in zip(pt_outputs, ms_outputs):
        if isinstance(p, ModelOutput):
            p = tuple(p.values())[0]

        p = p.detach().cpu().numpy()
        m = m.asnumpy()

        # relative error defined by Frobenius norm
        # dist(x, y) := ||x - y|| / ||y||, where ||·|| means Frobenius norm
        d = np.linalg.norm(p - m) / np.linalg.norm(p)

        diffs.append(d)

    return diffs