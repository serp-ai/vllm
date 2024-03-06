# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Sparsetral model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from vllm.transformers_utils.configs.sparsetral import SparsetralConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
    DEFAULT_VOCAB_PADDING_SIZE,
)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)
from vllm.sequence import SamplerOutput
from vllm.config import LoRAConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class ParallelAdapterMLP(nn.Module):
    def __init__(
        self,
        config,
        adapter_dim,
        adapter_scaling,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.adapter_down = RowParallelLinear(
            self.hidden_size, adapter_dim, bias=False, linear_method=linear_method
        )
        self.adapter_up = MergedColumnParallelLinear(
            adapter_dim, self.hidden_size, bias=False, linear_method=linear_method
        )
        self.adapter_act = nn.GELU()

        self.adapter_scaling = adapter_scaling

    def forward(self, x):
        x, _ = self.adapter_down(x)
        x = self.adapter_act(x)
        x, _ = self.adapter_up(x)
        return x * self.adapter_scaling


class SparsetralGateAdapter(nn.Module):
    def __init__(
        self, config: SparsetralConfig, linear_method: Optional[LinearMethodBase] = None
    ) -> None:
        super().__init__()

        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.topk = config.topk

        self.expert_indicies = list(range(self.num_experts))

        self.experts = nn.ModuleList(
            [
                ParallelAdapterMLP(
                    config,
                    config.adapter_dim,
                    config.moe_scaling,
                    linear_method=linear_method,
                )
                for _ in self.expert_indicies
            ]
        )

        self.router = RowParallelLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            linear_method=linear_method,
        )
        self.dtype = getattr(torch, config.moe_dtype)

    def forward(self, input_hidden_states, output_hidden_states, router_hidden_states):
        orig_shape = output_hidden_states.shape

        input_hidden_states = input_hidden_states.view(
            -1, input_hidden_states.shape[-1]
        )
        output_hidden_states = output_hidden_states.view(
            -1, output_hidden_states.shape[-1]
        )
        router_hidden_states = router_hidden_states.view(
            -1, router_hidden_states.shape[-1]
        )

        router_logits, _ = self.router(router_hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = None
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = selected_experts == expert_idx
            expert_weights = (routing_weights * expert_mask).sum(dim=-1, keepdim=True)

            current_hidden_states = (
                expert_layer(input_hidden_states)
                .add_(output_hidden_states)
                .mul_(expert_weights)
            )
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return final_hidden_states.view(*orig_shape)


class SparsetralMLP(nn.Module):
    def __init__(
        self,
        config: SparsetralConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            linear_method=linear_method,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False, linear_method=linear_method
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.moe_adapter = SparsetralGateAdapter(config, linear_method=linear_method)

    def forward(self, x):
        router_hidden_states = x
        gate_up, _ = self.gate_up_proj(x)
        up_proj = self.act_fn(gate_up)
        down_proj, _ = self.down_proj(up_proj)
        down_proj = self.moe_adapter(down_proj, down_proj, router_hidden_states)
        return down_proj


class SparsetralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        linear_method: Optional[LinearMethodBase] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class SparsetralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: SparsetralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = SparsetralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            linear_method=linear_method,
            sliding_window=config.sliding_window,
        )
        self.mlp = SparsetralMLP(
            config=config,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            linear_method=linear_method,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SparsetralModel(nn.Module):
    def __init__(
        self,
        config: SparsetralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList(
            [
                SparsetralDecoderLayer(config, linear_method)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SparsetralForCausalLM(nn.Module):
    # supports_lora = True

    def __init__(
        self,
        config: SparsetralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = SparsetralModel(config, linear_method, lora_config=lora_config)
        unpadded_vocab_size = config.vocab_size
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config
                else lora_config.lora_vocab_padding_size
            ),
        )
        self.sampler = Sampler(unpadded_vocab_size, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, sampling_metadata
        )
        return next_tokens

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = name.replace("expert_", "")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
