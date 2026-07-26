from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.models import granitemoe
from sglang.srt.models.gemma2 import Gemma2ForCausalLM
from sglang.srt.models.gemma2_reward import Gemma2ForSequenceClassification
from sglang.srt.models.internlm2 import InternLM2ForCausalLM
from sglang.srt.models.internlm2_reward import InternLM2ForRewardModel
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.llama_classification import LlamaForClassification
from sglang.srt.models.llama_reward import LlamaForSequenceClassification
from sglang.srt.models.mixtral import MixtralForCausalLM
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen2_classification import Qwen2ForSequenceClassification
from sglang.srt.models.qwen2_rm import Qwen2ForRewardModel


@pytest.mark.parametrize(
    ("wrapper_class", "base_class"),
    [
        (Gemma2ForSequenceClassification, Gemma2ForCausalLM),
        (InternLM2ForRewardModel, InternLM2ForCausalLM),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_reward_wrappers_select_base_implementation_directly(
    monkeypatch, wrapper_class, base_class, enabled
):
    calls = []

    def capture_legacy(module, weights):
        calls.append(("legacy", module, list(weights)))
        return {"legacy"}

    def capture_v2(module, weights):
        calls.append(("v2", module, list(weights)))
        return {"v2"}

    monkeypatch.setattr(base_class, "_legacy_load_weights", capture_legacy)
    monkeypatch.setattr(base_class, "_load_weights_v2", capture_v2)
    monkeypatch.setattr(
        base_class,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("unbound dispatcher was called"),
    )
    monkeypatch.setattr(
        envs,
        "SGLANG_ENABLE_WEIGHT_LOADER_V2",
        SimpleNamespace(get=lambda: enabled),
    )
    wrapper = wrapper_class.__new__(wrapper_class)
    torch.nn.Module.__init__(wrapper)
    weights = [("model.weight", torch.ones(1))]

    result = wrapper.load_weights(iter(weights))

    expected = "v2" if enabled else "legacy"
    assert result == {expected}
    assert calls == [(expected, wrapper, weights)]


@pytest.mark.parametrize(
    ("wrapper_class", "base_class"),
    [
        (LlamaForSequenceClassification, LlamaForCausalLM),
        (LlamaForClassification, LlamaForCausalLM),
        (Qwen2ForSequenceClassification, Qwen2ForCausalLM),
        (Qwen2ForRewardModel, Qwen2ForCausalLM),
    ],
)
def test_existing_wrappers_remain_on_direct_legacy_path(
    monkeypatch, wrapper_class, base_class
):
    calls = []

    def capture_legacy(module, weights):
        calls.append((module, list(weights)))
        return {"legacy"}

    monkeypatch.setattr(base_class, "_legacy_load_weights", capture_legacy)
    monkeypatch.setattr(
        base_class,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("unbound dispatcher was called"),
    )
    monkeypatch.setattr(
        base_class,
        "_load_weights_v2",
        lambda *_args, **_kwargs: pytest.fail("unsupported v2 loader was called"),
    )
    wrapper = wrapper_class.__new__(wrapper_class)
    torch.nn.Module.__init__(wrapper)
    weights = [("model.weight", torch.ones(1))]

    wrapper.load_weights(iter(weights))

    assert calls == [(wrapper, weights)]


def test_granitemoe_preserves_remap_and_uses_only_legacy_loader(monkeypatch):
    calls = []

    def capture_legacy(module, weights):
        calls.append((module, dict(weights)))
        return {"loaded"}

    monkeypatch.setattr(MixtralForCausalLM, "_legacy_load_weights", capture_legacy)
    monkeypatch.setattr(
        MixtralForCausalLM,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("unbound dispatcher was called"),
    )
    monkeypatch.setattr(
        MixtralForCausalLM,
        "_load_weights_v2",
        lambda *_args, **_kwargs: pytest.fail("unsupported v2 loader was called"),
    )
    wrapper = granitemoe.GraniteMoeForCausalLM.__new__(granitemoe.GraniteMoeForCausalLM)
    input_weight = torch.arange(16).reshape(2, 4, 2)
    output_weight = torch.arange(8).reshape(2, 2, 2)
    router_weight = torch.ones(2, 2)
    prefix = "model.layers.0.block_sparse_moe"

    result = wrapper.load_weights(
        [
            (f"{prefix}.input_linear.weight", input_weight),
            (f"{prefix}.output_linear.weight", output_weight),
            (f"{prefix}.router.layer.weight", router_weight),
        ]
    )

    assert result == {"loaded"}
    assert len(calls) == 1
    module, remapped = calls[0]
    assert module is wrapper
    for expert_id in range(2):
        expected_w1, expected_w3 = input_weight[expert_id].chunk(2, dim=0)
        torch.testing.assert_close(
            remapped[f"{prefix}.experts.{expert_id}.w1.weight"], expected_w1
        )
        torch.testing.assert_close(
            remapped[f"{prefix}.experts.{expert_id}.w3.weight"], expected_w3
        )
        torch.testing.assert_close(
            remapped[f"{prefix}.experts.{expert_id}.w2.weight"],
            output_weight[expert_id],
        )
    torch.testing.assert_close(remapped[f"{prefix}.gate.weight"], router_weight)
