import importlib
import logging

import mindspore
import numpy as np
import torch
from transformers import HubertConfig
from modeling_test_utils import compute_diffs, convert_state_dict

logger = logging.getLogger("ModelingsUnitTest")

THRESHOLD_FP32 = 5e-3
config = HubertConfig.from_pretrained("facebook/hubert-large-ls960-ft")
Hubert_test_cases = {
    "HubertLayerNormConvLayer": {
        'params': {
            'config': config
        },
        'input_shape': (1, 1, 93680)
    },
    "HubertFeatureEncoder": {
        'params': {
            'config': config
        },
        'input_shape': (1, 93680)
    },
    "HubertFeatureProjection": {
        'params': {
            'config': config
        },
        'input_shape': (1, 292, 512)
    },
    "HubertSamePadLayer": {
        'params': {
            'num_conv_pos_embeddings': config.num_conv_pos_embeddings
        },
        'input_shape': (1, 1024, 293)
    },
    "HubertPositionalConvEmbedding": {
        'params': {
            'config': config
        },
        'input_shape': (1, 1024, 293)
    },
    "HubertFeedForward": {
        'params': {
            'config': config
        },
        'input_shape': (1, 292, 1024)
    },
    "HubertEncoderLayerStableLayerNorm": {
        'params': {
            'config': config
        },
        'input_shape': (1, 292, 1024)
    },
    "HubertEncoderStableLayerNorm": {
        'params': {
            'config': config
        },
        'input_shape': (1, 292, 1024)
    }
}


def get_instance(class_path, *args, **kwargs):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(*args, **kwargs)


def sync_weights(pt_module_instance, ms_module_instance):
    missing_keys, unexpected_keys = mindspore.load_param_into_net(
        ms_module_instance, convert_state_dict(ms_module_instance, pt_module_instance.state_dict()), strict_load=True
    )
    if missing_keys or unexpected_keys:
        logger.warning(
            f"When load state_dict of '{pt_module}' to encounterpart mindspore model:\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}\n"
        )

    return pt_module_instance, ms_module_instance


target_modules = ['HubertAttention', 'HubertAttnAdapterLayer', 'HubertEncoder',
                    'HubertEncoderLayer', 'HubertEncoderLayerStableLayerNorm',
                    'HubertEncoderStableLayerNorm', 'HubertFeatureEncoder',
                    'HubertFeatureExtractor', 'HubertFeatureProjection', 'HubertFeedForward',
                    'HubertGroupNormConvLayer', 'HubertLayerNormConvLayer',
                    'HubertNoLayerNormConvLayer', 'HubertPositionalConvEmbedding',
                    'HubertSamePadLayer']

pt_prefix = 'transformers.models.hubert.modeling_hubert'
ms_prefix = 'mindone.models.hubert.modeling_hubert'


pt_modules = [pt_prefix + '.' + module for module in target_modules]
ms_modules = [ms_prefix + '.' + module for module in target_modules]

for pt_module, ms_module in zip(pt_modules, ms_modules):
    _, module_name = pt_module.rsplit('.', 1)
    if module_name in Hubert_test_cases:
        print(f"Module: {module_name}")
        init_args = ()
        init_kwargs = Hubert_test_cases[module_name]['params']
        intput_shape = Hubert_test_cases[module_name]['input_shape']

        pt_module_instance = get_instance(pt_module,  *init_args, **init_kwargs)
        ms_module_instance = get_instance(ms_module, *init_args, **init_kwargs)
        pt_module_instance, ms_module_instance = sync_weights(pt_module_instance, ms_module_instance)

        np_input = np.random.rand(*intput_shape).astype(np.float32)
        pt_input = torch.Tensor(np_input)
        ms_input = mindspore.Tensor(pt_input)

        pt_module_instance.eval()
        ms_module_instance.set_train(False)

        with torch.no_grad():
            pt_output = pt_module_instance(pt_input)
        ms_output = ms_module_instance(ms_input)

        diffs = compute_diffs(pt_output, ms_output)


        print(f"diff = {diffs}, If Valid: {(np.array(diffs) < THRESHOLD_FP32).all()}")
        print("-" * 50)
