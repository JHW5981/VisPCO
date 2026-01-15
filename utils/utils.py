import os
import torch
import torch.distributed as dist
import numpy as np
import random
import json
from pathlib import Path
import torch.nn as nn

def init_weights(m):
    """改进的权重初始化函数"""
    if isinstance(m, nn.Linear):
        # Xavier初始化
        nn.init.xavier_uniform_(m.weight)
        # 只在bias存在时初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # LayerNorm标准初始化
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
    else:    
        print("不是分布式训练...")
        args.distributed = False
        return
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
        print(f'| MASTER_ADDR: {os.environ['MASTER_ADDR']}| MASTER_PORT: {os.environ['MASTER_PORT']} | RANK: {args.rank} | WORLD_SIZE: {args.world_size} | LOCAL RANK: {args.gpu} |')
    else:
        print(f'| RANK: {args.rank} | WORLD_SIZE: {args.world_size} | LOCAL RANK: {args.gpu} |')
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url, 
        world_size=args.world_size, 
        rank=args.rank,
        device_id=torch.device(f'cuda:{args.gpu}')
        )
        
    torch.distributed.barrier()
    
    setup_for_distributed(args.rank == 0)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def load_pretrained_weight(model_path):
    if os.path.isdir(model_path):
        index_file = os.path.join(model_path, 'pytorch_model.bin.index.json')
        
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            shard_files = set(index_data['weight_map'].values())            
            # 加载所有分片并合并
            merged_state_dict = {}
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(model_path, shard_file)
                shard_weights = torch.load(shard_path, map_location='cpu')
                merged_state_dict.update(shard_weights)
            pretrained_weight = merged_state_dict
            
        else:
            model_files = []
            for file in os.listdir(model_path):
                if file.endswith(('.bin', '.pth', '.safetensors')):
                    model_files.append(file)
            if not model_files:
                raise FileNotFoundError(f"在文件夹 {model_path} 中未找到模型权重文件")

            if len(model_files) == 1:
                model_file = os.path.join(model_path, model_files[0])
                pretrained_weight = torch.load(model_file, map_location='cpu')
            else:
                pretrained_weight = {}
                for model_file in sorted(model_files):
                    file_path = os.path.join(model_path, model_file)
                    component_name = os.path.splitext(model_file)[0]
                    pretrained_weight[component_name] = torch.load(file_path, map_location='cpu')       
    else:
        pretrained_weight = torch.load(model_path, map_location='cpu')
    
    return pretrained_weight

def save_model(args, model, model_without_ddp, optimizer, loss_scaler, epoch, best_acc):
    output_dir = Path(args.save_ckpt_path)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'best_acc': best_acc,
        }
        save_on_master(to_save, checkpoint_path)

    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)

def batch_index_select_hidden_states(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        sorted_idx, _ = torch.sort(idx, dim=1)
        sorted_idx = sorted_idx + offset
        out = x.reshape(B*N, C)[sorted_idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        sorted_idx, _ = torch.sort(idx, dim=1)
        sorted_idx = sorted_idx + offset
        out = x.reshape(B*N)[sorted_idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_select_position_ids(x, idx):
    num_thw, B, N = x.size()
    _, N_new = idx.size()
    # 对索引进行排序，保持与其他函数一致的行为
    sorted_idx, _ = torch.sort(idx, dim=1)  # (B, N_new)
    
    # ============ 方法1：使用 torch.gather（推荐，最快）============
    # 将索引扩展到匹配 x 的第一个维度
    # sorted_idx: (B, N_new) -> (1, B, N_new) -> (num_thw, B, N_new)
    expanded_idx = sorted_idx.unsqueeze(0).expand(num_thw, B, N_new)
    
    # 使用 gather 在最后一个维度上选择
    out = torch.gather(x, dim=2, index=expanded_idx)  # (num_thw, B, N_new)
    
    return out


def batch_index_select_position_embeddings(x, idx):
    cos, sin = x
    num_thw, B, N, head_dim = cos.size()
    _, N_new = idx.size()
    
    # 对索引进行排序
    sorted_idx, _ = torch.sort(idx, dim=1)  # (B, N_new)
    
    # 扩展索引到匹配张量的维度
    # sorted_idx: (B, N_new) -> (1, B, N_new, 1) -> (num_thw, B, N_new, head_dim)
    expanded_idx = sorted_idx.unsqueeze(0).unsqueeze(-1).expand(num_thw, B, N_new, head_dim)
    
    # 使用 gather 在第2个维度（N维度）上选择
    cos_selected = torch.gather(cos, dim=2, index=expanded_idx)  # (num_thw, B, N_new, head_dim)
    sin_selected = torch.gather(sin, dim=2, index=expanded_idx)  # (num_thw, B, N_new, head_dim)
    
    return (cos_selected, sin_selected)

def batch_index_select_attention_mask(x, idx):
    if len(x.size()) != 4:
        raise ValueError(f"Expected x to have 4 dimensions, got {len(x.size())} dimensions")
    
    B, dim1, dim2, N = x.size()
    _, N_new = idx.size()
    
    # 对索引进行排序，保持与batch_index_select_hidden_states一致的行为
    sorted_idx, _ = torch.sort(idx, dim=1)
    
    # 检查是否是 (B, 1, 1, N) 的情况
    if dim1 == 1 and dim2 == 1:
        # 情况1: (B, 1, 1, N) -> (B, 1, 1, N_new)
        # 只需要在最后一个维度上进行选择
        x_squeezed = x.squeeze(1).squeeze(1)  # (B, N)
        
        # 使用gather在最后一个维度上选择
        selected = torch.gather(x_squeezed, dim=1, index=sorted_idx)  # (B, N_new)
        
        # 恢复维度
        out = selected.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N_new)
        
    else:
        # 情况2: (B, 1, N, N) -> (B, 1, N_new, N_new)
        # 原来的逻辑：同时选择行和列
        if dim1 != 1 or dim2 != N:
            raise ValueError(f"Expected x to be either (B, 1, 1, N) or (B, 1, N, N), got ({B}, {dim1}, {dim2}, {N})")
        
        # 移除中间维度进行处理
        x_squeezed = x.squeeze(1)  # (B, N, N)
        
        # 使用batch索引 + 高级索引同时选择行和列
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)  # (B, 1)
        
        # 先选择行: x_squeezed[batch_indices, sorted_idx, :]
        # 形状变为 (B, N_new, N)
        selected_rows = x_squeezed[batch_indices, sorted_idx, :]
        
        # 再选择列: selected_rows[:, :, sorted_idx]
        # 需要用gather来选择列，因为sorted_idx的形状是(B, N_new)
        sorted_idx_expanded = sorted_idx.unsqueeze(1).expand(B, N_new, N_new)
        selected_mask = torch.gather(selected_rows, dim=2, index=sorted_idx_expanded)
        
        # 恢复中间维度
        out = selected_mask.unsqueeze(1)  # (B, 1, N_new, N_new)
    
    return out