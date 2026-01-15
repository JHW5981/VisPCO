import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import transformers
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging, can_return_tuple, is_torchdynamo_compiling
from transformers.generation import GenerationMixin

from models.qwen2_5vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from models.qwen2_5vl.vision_encoder import Qwen2_5_VisionTransformerPretrainedModel
from models.qwen2_5vl.modeling_utils import Qwen2_5_VLPreTrainedModel, Qwen2RMSNorm
from utils.utils import (batch_index_select_hidden_states, batch_index_select_attention_mask,
                   batch_index_select_position_ids, batch_index_select_position_embeddings)


logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    
    assert cos.shape[2] == q.shape[2]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def softmax_with_policy(attn: torch.Tensor, policy: torch.Tensor, eps=1e-6) -> torch.Tensor:
    B, N, _ = policy.size()
    B, H, N, N = attn.size()
    attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
    eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
    attn_policy = attn_policy + (1.0 - attn_policy) * eye
    max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    attn = attn - max_att
    # attn = attn.exp_() * attn_policy
    # return attn / attn.sum(dim=-1, keepdim=True)

    # for stable training
    attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
    attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
    return attn.type_as(max_att)

def softmax_manual(attn: torch.Tensor, dim: int = -1, dtype=None, eps=1e-6) -> torch.Tensor:
    """
    手动实现 softmax（仿 nn.functional.softmax，但不用底层加速）
    支持:
        - 指定 dim
        - 指定 dtype
        - 数值稳定 (减去 max)
    """
    if dtype is not None:
        attn = attn.to(dtype)

    # 数值稳定性：减去每个slice的最大值
    max_attn = torch.max(attn, dim=dim, keepdim=True)[0]
    attn_exp = torch.exp(attn - max_attn)
    attn_sum = torch.sum(attn_exp, dim=dim, keepdim=True)

    # 为避免除零，加上极小值 eps
    softmax_out = attn_exp / (attn_sum + eps)
    return softmax_out

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    policy: Optional[torch.Tensor] = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] # BUG: 这里有个bug，只有在evaluation的batch是1的时候，attn_mask全为1的时候正常
        attn_weights = attn_weights + causal_mask

    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query.dtype)
        if policy is None:
            # 为了公平比较时间，去掉原来softmax的编译加速
            attn_weights = softmax_manual(attn_weights, dim=-1, dtype=query.dtype)      
        else:
            attn_weights = softmax_with_policy(attn_weights, policy).to(query.dtype)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    out_pred_prob: Optional[torch.IntTensor] = None
    flop_loss: Optional[torch.FloatTensor] = None

@dataclass
class Qwen2_5_VLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    out_pred_prob: Optional[torch.IntTensor] = None
    flop_loss: Optional[torch.FloatTensor] = None

class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        policy: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            policy=policy,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        policy: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            policy=policy,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class ResidualBlock(nn.Module):
    """带残差连接的基础块"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 如果输入输出维度不同,使用投影
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return self.net(x) + self.proj(x)

# 挖掘输入的复杂性，根据给定的budget，预测剪枝参数
class PredictPruningRatio(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size // 2)
        
        self.feature_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_size // 2, self.hidden_size // 4, dropout=0.1),
            ResidualBlock(self.hidden_size // 4, self.hidden_size // 8, dropout=0.1),
        ])
        
        # Budget编码器
        self.budget_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, self.hidden_size // 4)
        )
        
        # ✅ 关键修复：FiLM参数应该匹配block的**输出**维度
        # block输出维度 * 2 (因为要chunk成scale和shift)
        self.film_layers = nn.ModuleList([
            nn.Linear(self.hidden_size // 4, self.hidden_size // 4 * 2),  # for first block output (H//4)
            nn.Linear(self.hidden_size // 4, self.hidden_size // 8 * 2),  # for second block output (H//8)
        ])
        
        self.predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_size // 8),
            nn.Linear(self.hidden_size // 8, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, hidden_states, budget, visual_mask, attention_mask):      
        B = hidden_states.size(0)
        
        # 计算budget范围并归一化
        max_budgets, min_budgets = self.compute_budget_range(visual_mask, attention_mask)
        max_budgets = max_budgets.to(dtype=hidden_states.dtype)
        min_budgets = min_budgets.to(dtype=hidden_states.dtype)
        target_budget = min_budgets + (max_budgets - min_budgets) * budget
        normalized_budget = self.normalize_budget(target_budget, max_budgets, min_budgets).to(dtype=hidden_states.dtype)  # (B,)
        
        # 特征提取
        x = self.input_proj(hidden_states)  # (B, N, H//2)
        budget_emb = self.budget_encoder(normalized_budget.unsqueeze(-1))  # (B, H//4)
        
        # 通过特征块,使用FiLM调制
        for i, block in enumerate(self.feature_blocks):
            # 先过block
            x = block(x)  # (B, N, output_dim)
            # 第0个block: (B, N, H//4)
            # 第1个block: (B, N, H//8)
            
            # 生成对应输出维度的scale和shift参数
            film_params = self.film_layers[i](budget_emb)  # (B, output_dim * 2)
            scale, shift = film_params.chunk(2, dim=-1)  # 各 (B, output_dim)
            
            # FiLM调制: x = scale * x + shift
            scale = scale.unsqueeze(1)  # (B, 1, output_dim)
            shift = shift.unsqueeze(1)  # (B, 1, output_dim)
            x = scale * x + shift
        
        x_pooled = x.mean(dim=1)  # (B, H//8)
        predictions = self.predictor(x_pooled)  # (B, 2)
        
        # 解析预测结果
        k_logits = predictions[:, 0]  # (B,)
        r_logits = predictions[:, 1]  # (B,)
        
        # 转换为实际的k和r值
        k = torch.sigmoid(k_logits) * self.num_hidden_layers  # (B,) 范围[0, num_layers]
        r = torch.sigmoid(r_logits)  # (B,) 范围[0.1, 1]
        
        final_r = self.kernel_func(k, r)
        
        # if self.training:
        #     flop_loss = self.get_budget_constraint_loss(
        #         final_r, visual_mask, attention_mask, hidden_states, target_budget
        #     )
        # else:
        #     flop_loss = None

        flop_loss = self.get_budget_constraint_loss(
            final_r, visual_mask, attention_mask, hidden_states, target_budget
        ) 
        return final_r, flop_loss

    def compute_flops(self, num_tokens):
        """
        更准确的FLOPs计算
        Args:
            num_tokens: token数量
        Returns:
            每层的FLOPs
        """
        # Attention的FLOPs计算
        # QKV projection: 3 * N * H * H
        qkv_flops = 3 * num_tokens * self.hidden_size * self.hidden_size
        
        # Attention scores: N * N * H (QK^T)
        attn_score_flops = num_tokens * num_tokens * self.hidden_size
        
        # Attention output: N * N * H (Attention * V)
        attn_output_flops = num_tokens * num_tokens * self.hidden_size
        
        # Output projection: N * H * H
        output_proj_flops = num_tokens * self.hidden_size * self.hidden_size
        
        total_attn_flops = qkv_flops + attn_score_flops + attn_output_flops + output_proj_flops
        
        # FeedForward的FLOPs计算
        # 两层MLP: N * H * I + N * I * H
        ff_flops = 2 * num_tokens * self.hidden_size * self.intermediate_size
        
        # 每层总FLOPs
        layer_flops = total_attn_flops + ff_flops
        
        return layer_flops / 1e12  # 返回TFLOPs

    def compute_budget_range(self, visual_mask, attention_mask):
        B, N = visual_mask.size()
        
        # 计算每个样本的最大budget(不剪枝)
        max_budgets = []
        min_budgets = []
        
        for i in range(B):
            # 最大budget: 所有token都保留
            N = attention_mask[i].sum().item()
            layer_flops_max = self.compute_flops(N)
            max_budget = self.num_hidden_layers * layer_flops_max
            max_budgets.append(max_budget)
            
            # 最小budget: 只保留文本token
            num_visual = visual_mask[i].sum().item()
            num_text = N - num_visual
            layer_flops_min = self.compute_flops(num_text)
            min_budget = self.num_hidden_layers * layer_flops_min
            min_budgets.append(min_budget)
        
        max_budgets = torch.tensor(max_budgets, device=visual_mask.device, dtype=torch.bfloat16)
        min_budgets = torch.tensor(min_budgets, device=visual_mask.device, dtype=torch.bfloat16)
        
        return max_budgets, min_budgets

    def normalize_budget(self, budget, max_budgets, min_budgets):
        """
        将budget归一化到[0, 1]范围
        Args:
            budget: 标量或(B,)的tensor
            max_budgets: (B,)
            min_budgets: (B,)
        Returns:
            normalized_budget: (B,)
        """
        B = max_budgets.size(0)
        
        # 如果budget是标量,扩展为(B,)
        if isinstance(budget, (int, float)):
            budget = torch.full((B,), budget, device=max_budgets.device, dtype=max_budgets.dtype)
        elif budget.dim() == 0:
            budget = budget.unsqueeze(0).expand(B).to(dtype=max_budgets.dtype)
        # 限制budget在有效范围内
        budget = torch.clamp(budget, min_budgets, max_budgets).to(dtype=max_budgets.dtype)
        
        # 归一化
        budget_range = max_budgets - min_budgets
        # 避免除零
        budget_range = torch.where(budget_range > 0, budget_range, torch.ones_like(budget_range))
        normalized_budget = (budget - min_budgets) / budget_range
        
        return normalized_budget

    def kernel_func(self, k, r):
        r_list = []
        for i in range(self.num_hidden_layers):
            x = 10 * (i - k)
            s = torch.sigmoid(x)
            r_i = 1 + (r - 1) * s
            r_list.append(r_i)

        return torch.stack(r_list, dim=1)

    def compute_actual_budget(self, r, visual_mask, attention_mask, hidden_states):
        B, N, H = hidden_states.size()
        actual_budgets = []
        
        for i in range(B):
            N = attention_mask[i].sum().item()
            total_flops = 0
            num_visual = visual_mask[i].sum().item()
            num_text = N - num_visual
            
            for layer_idx in range(self.num_hidden_layers):
                r_i = r[i][layer_idx]
                effective_tokens = num_text + num_visual * r_i
                
                # 计算这一层的FLOPs
                if effective_tokens > 0:
                    layer_flops = self.compute_flops(effective_tokens)
                    total_flops += layer_flops
            
            actual_budgets.append(total_flops)
        
        actual_budgets = torch.stack(actual_budgets, dim=0).to(dtype=hidden_states.dtype)
        return actual_budgets

    def get_budget_constraint_loss(self, r, visual_mask, attention_mask, hidden_states, target_budget):
        # 计算实际budget
        actual_budgets = self.compute_actual_budget(r, visual_mask, attention_mask, hidden_states)
        # loss = F.mse_loss(actual_budgets, target_budget).to(actual_budgets.dtype)
        loss = (target_budget - actual_budgets).sum()
        return loss

class Qwen2_5_VLTextModel(Qwen2_5_VLPreTrainedModel):
    config_class = Qwen2_5_VLTextConfig

    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__(config)
        # 把attention的类型设置成eager
        self.config._attn_implementation = "eager"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.predict_pruning_ratio = PredictPruningRatio(config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.gradient_checkpointing = False

        self.r = None

        # self.pruning_loc = config.pruning_loc
        self.pruning_loc = list[int](range(1,len(self.layers)+1))
        # self.token_ratio = config.token_ratio
        self.keep_policy_list = []
        self.token_ratio = []
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        policy_mask: torch.LongTensor = None,
        budget: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        p_count = 0
        out_pred_prob = []
        init_n = seq_length
        B = batch_size
        prev_decision = attention_mask.unsqueeze(2).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        policy = attention_mask.unsqueeze(2).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        
        if inputs_embeds.shape[1] > 1: 
            self.r, self.flop_loss = self.predict_pruning_ratio(hidden_states, budget, policy_mask, attention_mask)
        else: # FIXME: 这里比较粗糙的处理了，并不严谨，表示是decode阶段
            pass
        
        for idx, decoder_layer in enumerate(self.layers):
            if self.training:
                token_ratio = self.r[:, idx]
            else:
                if idx == 0:
                    token_ratio = self.r[:, idx]
                else:
                    token_ratio = self.r[:, idx] / self.r[:, idx - 1]
            if idx == 0:
                self.token_ratio = []
            self.token_ratio.append(token_ratio.detach().cpu().tolist())

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if idx+1 in self.pruning_loc:
                output_attentions_2 = True
            else:
                output_attentions_2 = False | output_attentions
                
            if idx in self.pruning_loc:
                # compute pruned tokens, generate fastv sign
                last_layer_attention = layer_outputs[1]
                # compute average attention over different head
                last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)
                # generate new attention mask based on the average attention, sample the top ATTENTION_RANK tokens with highest attention
                pred_score = last_layer_attention_avg[:, -1, :]
                pred_score = (pred_score - pred_score.min(dim=1, keepdim=True)[0]) / (pred_score.max(dim=1, keepdim=True)[0] - pred_score.min(dim=1, keepdim=True)[0] + 1e-8)
               
                if self.training:
                    score = pred_score * policy_mask + (1 - policy_mask) * 1000
                    score = torch.log(score + 1e-8)
                    # image_score = score * policy_mask
                    sorted_score, _ = torch.sort(score, dim=1, descending=True)
                    num_keep_node_float = policy_mask.sum(dim=1) * token_ratio + (1 - policy_mask).sum(dim=1)
                    # 使用 num_keep_node_float（带梯度）来插值阈值
                    B = policy_mask.shape[0]
                    N = policy_mask.shape[1]
                    cumsum_mask = torch.arange(N, device=sorted_score.device).unsqueeze(0).expand(B, -1).float()
                    soft_k_position = num_keep_node_float.unsqueeze(1)  # 保持梯度
                    
                    # 软插值找阈值（可导）
                    position_weights = torch.exp(-((cumsum_mask - soft_k_position)**2) / 5).to(soft_k_position.dtype) # 5是个超参数，越小越尖锐，越集中，阈值越准，但是容易出现训练不稳定
                    soft_thresholds = (sorted_score * position_weights).sum(dim=1, keepdim=True) / (position_weights.sum(dim=1, keepdim=True) + 1e-8)
                           
                    # 选择阈值
                    thresholds = soft_thresholds
                    # Hard mask
                    hard_keep_decision = (score >= thresholds).to(dtype=hidden_states.dtype)
                    
                    # Soft mask (STE)
                    temperature = 0.1 # 超参数
                    soft_mask = torch.sigmoid((score - thresholds) / temperature)
                    hard_keep_decision = soft_mask + (hard_keep_decision - soft_mask).detach()
                    hard_keep_decision = hard_keep_decision.unsqueeze(-1) * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    policy = hard_keep_decision.reshape(B, init_n, 1)

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions_2,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        policy=policy,
                        **kwargs,
                    )
                    prev_decision = hard_keep_decision
                else:
                    if inputs_embeds.shape[1] > 1: # BUG: 这里比较粗糙的处理了，并不严谨，表示是prefill阶段
                        if p_count == 0: # 确保每次eval的时候都重新记录
                            self.keep_policy_list = []
                        score = pred_score * policy_mask + (1 - policy_mask) * 1000
                        score = torch.log(score + 1e-8)
                        num_keep_node = int(policy_mask.sum().item() * token_ratio + (1 - policy_mask).sum().item())
                        keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                        hidden_states = batch_index_select_hidden_states(hidden_states, keep_policy)
                        position_ids = batch_index_select_position_ids(position_ids, keep_policy)
                        prev_decision = batch_index_select_hidden_states(prev_decision, keep_policy)
                        policy_mask = batch_index_select_hidden_states(policy_mask, keep_policy)
                        causal_mask_mapping[decoder_layer.attention_type] = batch_index_select_attention_mask(causal_mask_mapping[decoder_layer.attention_type], keep_policy)
                        position_embeddings = batch_index_select_position_embeddings(position_embeddings, keep_policy)
                        cache_position = cache_position[keep_policy.sort()[0]][0]
                        self.keep_policy_list.append(keep_policy)
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions_2,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )
                    else: # decode阶段
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions_2,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            **kwargs,
                        )     
                p_count += 1
                output_attentions_2 = True
            else:
                if self.training:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions_2,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        policy=policy,
                        **kwargs,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions_2,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions_2 else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            out_pred_prob=out_pred_prob if self.training else None,
            flop_loss=self.flop_loss if self.training else None,
        )

class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    ## normalize type, send to device.
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        budget: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mask = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0)
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            policy_mask=mask.int() if mask is not None else None, # mask是None的情况只能是generate的decode阶段
            budget=budget,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            out_pred_prob=outputs.out_pred_prob if self.training else None,
            flop_loss=outputs.flop_loss if self.training else None,
        )
        return output if return_dict else output.to_tuple()


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    label_logits: Optional[torch.FloatTensor] = None
    out_pred_prob: Optional[torch.IntTensor] = None
    flop_loss: Optional[torch.FloatTensor] = None


class Qwen2_5_VLForConditionalGeneration_FastV(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.finetuning = False
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        budget: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 为了兼容calc_flops在这里改动吧 # FIXME
        if isinstance(input_ids, transformers.feature_extraction_utils.BatchFeature) or isinstance(input_ids, dict):
            # labels = input_ids['labels']
            pixel_values = input_ids['pixel_values']
            attention_mask = input_ids['attention_mask']
            image_grid_thw = input_ids['image_grid_thw']
            position_ids = input_ids['position_ids'] if 'position_ids' in input_ids else None # 计算tflops的时候没有这个
            budget = input_ids['budget']
            input_ids = input_ids['input_ids']


        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            budget=budget,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        label_logits = None
        if labels is not None and self.finetuning:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        else:
            label_logits = logits[labels != -100]

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            label_logits=label_logits if labels is not None and not self.finetuning else None, # 只有在蒸馏训练的时候才有
            out_pred_prob=outputs.out_pred_prob if self.training else None,
            flop_loss=outputs.flop_loss if self.training else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        budget=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        model_inputs["budget"] = budget
        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = ["Qwen2_5_VLForConditionalGeneration_FastV", "Qwen2_5_VLModel", "Qwen2_5_VLTextModel"]


