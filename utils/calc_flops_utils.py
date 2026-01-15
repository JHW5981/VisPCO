import torch
import torch.nn as nn
from models.qwen2_5vl.modeling_utils import Qwen2RMSNorm
from models.qwen2_5vl.vision_encoder import Qwen2_5_VLVisionAttention
from models.qwen2_5vl.qwen2_5_vl_fastv import Qwen2_5_VLAttention

def count_qwen2_rmsnorm(m, x, y):
    input = y[0]  # 使用输出
    
    # pow(2) + mean + rsqrt + 2个逐元素乘法
    # 计算总元素数
    numel = input.numel()
    total_flops = 5 * numel
    
    m.total_ops += torch.DoubleTensor([total_flops]).to(m.total_ops.device)

def count_qwen_vision_attention(m, x, y):
    """
    统计 Qwen2_5_VLVisionAttention 的 Attention 核心计算 FLOPs
    Linear 层 (qkv, proj) 应该已被 Linear 规则统计
    这里只统计 QK^T + softmax + AV
    """
    hidden_states = y[0]  # 输入 (seq_len, hidden_size)
    seq_len = hidden_states.shape[0]
    
    num_heads = m.num_heads
    head_dim = m.head_dim
    
    # Attention 核心计算
    # QK^T: (H, L, D) x (H, D, L) -> (H, L, L)
    qk_flops = num_heads * seq_len * seq_len * head_dim * 2
    
    # softmax (粗略估计)
    softmax_flops = num_heads * seq_len * seq_len * 5
    
    # AV: (H, L, L) x (H, L, D) -> (H, L, D)
    av_flops = num_heads * seq_len * seq_len * head_dim * 2
    
    total_flops = qk_flops + softmax_flops + av_flops
    
    m.total_ops += torch.DoubleTensor([total_flops]).to(m.total_ops.device)

def count_qwen_text_attention(m, x, y):
    """
    统计 Qwen2_5_VLAttention 的 Attention 核心计算 FLOPs (支持 GQA)
    Linear 层 (q_proj, k_proj, v_proj, o_proj) 应该已被 Linear 规则统计
    这里只统计 QK^T + softmax + AV
    """
    hidden_states = y[0]  # 输入 (batch_size, seq_len, hidden_size)
    bsz, q_len, _ = hidden_states.shape
    
    num_heads = m.num_heads
    head_dim = m.head_dim
    
    # 对于 GQA，kv_seq_len 可能不同（如果有 cache），这里假设等于 q_len
    # 实际推理时如果有 past_key_value，kv_seq_len 会更大，但这里统计的是当前步的计算
    kv_seq_len = q_len
    
    # Attention 核心计算（对于 GQA，每个 query head 都要计算）
    # QK^T: (B, H, L_q, D) x (B, H, D, L_kv) -> (B, H, L_q, L_kv)
    qk_flops = bsz * num_heads * q_len * kv_seq_len * head_dim * 2
    
    # softmax (粗略估计)
    softmax_flops = bsz * num_heads * q_len * kv_seq_len * 5
    
    # AV: (B, H, L_q, L_kv) x (B, H, L_kv, D) -> (B, H, L_q, D)
    av_flops = bsz * num_heads * q_len * kv_seq_len * head_dim * 2
    
    total_flops = qk_flops + softmax_flops + av_flops
    
    m.total_ops += torch.DoubleTensor([total_flops]).to(m.total_ops.device)

custom_ops = {}
custom_ops[Qwen2RMSNorm] = count_qwen2_rmsnorm
custom_ops[Qwen2_5_VLVisionAttention] = count_qwen_vision_attention
custom_ops[Qwen2_5_VLAttention] = count_qwen_text_attention

__all__ = ['custom_ops']
