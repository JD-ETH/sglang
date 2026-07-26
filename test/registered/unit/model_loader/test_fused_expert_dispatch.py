"""CPU unit tests for FusedExpertDispatch expert fan-out."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch.nn import Parameter

from sglang.srt.model_loader.auto_loader import (
    ExpertParamsDispatch,
    FusedExpertDispatch,
    load_qwen35_moe_checkpoint_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFusedExpertDispatch(unittest.TestCase):
    def test_fan_out_gate_up_splits_w1_w3(self):
        num_experts = 3
        calls = []

        def weight_loader(param, shard_tensor, runtime_name, shard_id, expert_id):
            calls.append((shard_id, expert_id, shard_tensor.shape))

        param = MagicMock(spec=Parameter)
        param.weight_loader = weight_loader

        gate = torch.randn(num_experts, 4, 4)
        up = torch.randn(num_experts, 4, 4)
        fused = torch.cat([gate, up], dim=-2)

        params = {
            "model.layers.0.mlp.experts.w13_weight.weight": param,
        }
        dispatch = FusedExpertDispatch(num_experts=num_experts)
        ckpt_name = "model.layers.0.mlp.experts.gate_up_proj.weight"

        loaded = dispatch.try_load(ckpt_name, fused, params)
        self.assertEqual(loaded, "model.layers.0.mlp.experts.w13_weight.weight")
        self.assertEqual(len(calls), num_experts * 2)

    def test_fan_out_down_proj(self):
        num_experts = 2
        calls = []

        def weight_loader(param, shard_tensor, runtime_name, shard_id, expert_id):
            calls.append((shard_id, expert_id))

        param = MagicMock(spec=Parameter)
        param.weight_loader = weight_loader

        fused = torch.randn(num_experts, 6, 3)
        params = {"layer.mlp.experts.w2_weight.weight": param}
        dispatch = FusedExpertDispatch(num_experts=num_experts)

        loaded = dispatch.try_load("layer.mlp.experts.down_proj.weight", fused, params)
        self.assertEqual(loaded, "layer.mlp.experts.w2_weight.weight")
        self.assertEqual(calls, [("w2", 0), ("w2", 1)])

    def test_static_fan_out_helper(self):
        calls = []

        def weight_loader(param, shard_tensor, runtime_name, shard_id, expert_id):
            calls.append(expert_id)

        param = MagicMock(spec=Parameter)
        param.weight_loader = weight_loader
        tensor = torch.randn(4, 2, 2)

        FusedExpertDispatch.fan_out_to_experts(
            param, tensor, "experts.w13_weight", "w1", 4
        )
        self.assertEqual(calls, [0, 1, 2, 3])


class TestQwen35CheckpointRouting(unittest.TestCase):
    def test_visual_mlp_weight_bypasses_text_stacked_dispatch(self):
        loaded = []
        param = Parameter(torch.zeros(2, 2))
        param.weight_loader = lambda _param, tensor: loaded.append(tensor)
        runtime_name = "visual.mlp.gate_proj.weight"
        module = SimpleNamespace(
            named_parameters=lambda remove_duplicate=False: [(runtime_name, param)]
        )
        tensor = torch.ones(2, 2)

        result = load_qwen35_moe_checkpoint_weights(
            module,
            [("model.visual.mlp.gate_proj.weight", tensor)],
            num_experts=1,
            expert_dispatch=ExpertParamsDispatch(),
            fused_dispatch=None,
            skip_substrs=(),
            remap_visual=True,
        )

        self.assertEqual(result, {runtime_name})
        self.assertEqual(len(loaded), 1)
        torch.testing.assert_close(loaded[0], tensor)

    def test_encoder_only_skips_all_expert_dispatch(self):
        calls = []

        def weight_loader(param, tensor, qualname, shard_id=None, expert_id=None):
            calls.append((qualname, shard_id, expert_id))

        param = Parameter(torch.zeros(1))
        param.weight_loader = weight_loader
        runtime_name = "model.layers.0.mlp.experts.w13_weight"
        module = SimpleNamespace(
            named_parameters=lambda remove_duplicate=False: [(runtime_name, param)]
        )
        dispatch = ExpertParamsDispatch(
            mappings=(("experts.w13_", "experts.0.gate_proj.", 0, "w1"),)
        )

        result = load_qwen35_moe_checkpoint_weights(
            module,
            [("model.layers.0.mlp.experts.0.gate_proj.weight", torch.ones(1))],
            num_experts=1,
            expert_dispatch=dispatch,
            fused_dispatch=None,
            skip_substrs=(),
            encoder_only=True,
        )

        self.assertEqual(result, set())
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
