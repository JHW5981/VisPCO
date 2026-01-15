import torch
import math
import warnings
import logging
import os

# 关闭 transformers 关于 generation flags 的警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*generation flags.*')
warnings.filterwarnings('ignore', message='.*The following generation flags.*')
warnings.filterwarnings('ignore', message='.*may be ignored.*')
# 设置 transformers logger 级别
logging.getLogger('transformers.generation.utils').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

from utils.logger import MetricLogger, SmoothedValue
from tqdm import tqdm
from eval.eval_utils import calculate_choice_accuracy, calculate_anls, calculate_relaxed_accuracy, calculate_yes_no_accuracy

precision_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def calculate_metric(generated_answers, gt_answer, dataset_eval_name, metric_logger):
    assert len(generated_answers) == len(gt_answer)
    accuracy = None
    relaxed_accuracy = None
    # 选择题 - 使用选项提取后的准确率
    if dataset_eval_name in ['A-OKVQA', 'CCBench', 'MMBench_DEV_CN', 'MMMU_DEV_VAL', 'ScienceQA_VAL', 'SEEDBench_IMG']:
        accuracy = calculate_choice_accuracy(generated_answers, gt_answer)
        metric_logger.meters[f'{dataset_eval_name}_Accuracy'].update(accuracy, n=len(gt_answer))
    # 简短问答 - 使用ANLS和宽松准确率
    elif dataset_eval_name in ['ChartQA', 'OCRBench', 'VizWiz', 'TextVQA_VAL', 'tablevqa-bench']: # VizWiz的answer是个列表
        # anls = calculate_anls(generated_answers, gt_answer, threshold=0.5)
        relaxed_accuracy = calculate_relaxed_accuracy(generated_answers, gt_answer)
        # metric_logger.meters[f'{dataset_eval_name}_ANLS'].update(anls, n=len(gt_answer))
        metric_logger.meters[f'{dataset_eval_name}_Relaxed_Accuracy'].update(relaxed_accuracy, n=len(gt_answer))
    # 长问答
    elif dataset_eval_name in ['LLaVABench', 'MM-Math']:
        pass
    # Yes/No 问答
    elif dataset_eval_name in ['MME']:
        accuracy = calculate_yes_no_accuracy(generated_answers, gt_answer)
        metric_logger.meters[f'{dataset_eval_name}_Accuracy'].update(accuracy, n=len(gt_answer))
    else:
        raise ValueError(f"Unknown dataset: {dataset_eval_name}")
    result = accuracy if accuracy is not None else relaxed_accuracy
    # print("===================================", flush=True)
    # print("标准答案：", gt_answer, flush=True)
    # print("生成答案：", generated_answers, flush=True)
    # print("预测结果：", result, flush=True)
    # print("===================================", flush=True)
    return result


def compute_flops(num_tokens, hidden_size, intermediate_size):
    """
    更准确的FLOPs计算
    Args:
        num_tokens: token数量
    Returns:
        每层的FLOPs
    """
    # Attention的FLOPs计算
    # QKV projection: 3 * N * H * H
    qkv_flops = 3 * num_tokens * hidden_size * hidden_size
    
    # Attention scores: N * N * H (QK^T)
    attn_score_flops = num_tokens * num_tokens * hidden_size
    
    # Attention output: N * N * H (Attention * V)
    attn_output_flops = num_tokens * num_tokens * hidden_size
    
    # Output projection: N * H * H
    output_proj_flops = num_tokens * hidden_size * hidden_size
    
    total_attn_flops = qkv_flops + attn_score_flops + attn_output_flops + output_proj_flops
    
    # FeedForward的FLOPs计算
    # 两层MLP: N * H * I + N * I * H
    ff_flops = 2 * num_tokens * hidden_size * intermediate_size
    
    # 每层总FLOPs
    layer_flops = total_attn_flops + ff_flops
    
    return layer_flops / 1e12  # 返回TFLOPs


def compute_actual_budget(num_hidden_layers, hidden_size, intermediate_size, r, visual_mask, attention_mask):
    B, N = attention_mask.size()
    actual_budgets = []
    
    for i in range(B):
        N = attention_mask[i].sum().item()
        total_flops = 0
        num_visual = visual_mask[i].sum().item()
        num_text = N - num_visual
        
        for layer_idx in range(num_hidden_layers):
            r_i = r[i][layer_idx]
            effective_tokens = num_text + num_visual * r_i
            
            # 计算这一层的FLOPs
            if effective_tokens > 0:
                layer_flops = compute_flops(effective_tokens, hidden_size, intermediate_size)
                total_flops += layer_flops
        
        actual_budgets.append(total_flops)
    
    actual_budgets = torch.stack(actual_budgets, dim=0)
    return actual_budgets

def compute_budget_range(num_hidden_layers, hidden_size, intermediate_size, visual_mask, attention_mask):
    B, N = visual_mask.size()
    
    # 计算每个样本的最大budget(不剪枝)
    max_budgets = []
    min_budgets = []
    
    for i in range(B):
        # 最大budget: 所有token都保留
        N = attention_mask[i].sum().item()
        layer_flops_max = compute_flops(N, hidden_size, intermediate_size)
        max_budget = num_hidden_layers * layer_flops_max
        max_budgets.append(max_budget)
        
        # 最小budget: 只保留文本token
        num_visual = visual_mask[i].sum().item()
        num_text = N - num_visual
        layer_flops_min = compute_flops(num_text, hidden_size, intermediate_size)
        min_budget = num_hidden_layers * layer_flops_min
        min_budgets.append(min_budget)
    
    max_budgets = torch.tensor(max_budgets, device=visual_mask.device, dtype=torch.bfloat16)
    min_budgets = torch.tensor(min_budgets, device=visual_mask.device, dtype=torch.bfloat16)
    
    return max_budgets, min_budgets

@torch.no_grad()
def evaluate(eval_dataloader_lists, dataset_eval_names, model, processor, args, ratio):
    # 获取原始模型（处理 DDP 包装）
    from torch.nn.parallel import DistributedDataParallel as DDP
    model_for_inference = model.module if isinstance(model, DDP) else model
    model_for_inference.eval()
    
    metric_logger = MetricLogger(delimiter=" ")
    header = 'Test:'
    
    for eval_dataloader, dataset_eval_name in tqdm(zip(eval_dataloader_lists, dataset_eval_names)):
        print(f"Evaluating {dataset_eval_name}...", flush=True)

        for batch in metric_logger.log_every(eval_dataloader, 10, header):
            current_num = len(batch['input_ids'])
            batch['input_ids'] = batch['input_ids'].to(args.device)
            dtype = precision_map[args.precision]
            batch['attention_mask'] = batch['attention_mask'].to(args.device)
            batch['image_grid_thw'] = batch['image_grid_thw'].to(args.device)
            batch['position_ids'] = batch['position_ids'].to(args.device)
            batch['pixel_values'] = batch['pixel_values'].to(args.device, dtype=dtype)
            gt_answer = batch['answers']
            batch['budget'] = torch.full((batch['input_ids'].shape[0],), ratio, device=batch['input_ids'].device, dtype=dtype)
            # 使用原始模型进行生成
            generated_answers = model_for_inference.generate(
                inputs=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                position_ids=batch['position_ids'],
                budget=batch['budget'],
                do_sample=False,
                # temperature=0.2,
                # top_p=None,
                # num_beams=1,
                max_new_tokens=64,
                use_cache=True,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch['input_ids'], generated_answers)
            ]

            generated_answers = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            metric = calculate_metric(generated_answers, gt_answer, dataset_eval_name, metric_logger)
            # metric_logger.meters[f'{dataset_eval_name}_Metric'].update(metric, n=current_num)
            r = model_for_inference.model.language_model.r
            flops = compute_actual_budget(model_for_inference.model.config.num_hidden_layers, model_for_inference.model.config.hidden_size, model_for_inference.model.config.intermediate_size, r, batch['input_ids'] == 151655, batch['attention_mask'])
            metric_logger.update(flops=flops.item())
            max_budgets, min_budgets = compute_budget_range(model_for_inference.model.config.num_hidden_layers, model_for_inference.model.config.hidden_size, model_for_inference.model.config.intermediate_size, batch['input_ids'] == 151655, batch['attention_mask'])
            metric_logger.update(max_budgets=max_budgets.item())
            metric_logger.update(min_budgets=min_budgets.item())

            if args.distributed:
                torch.distributed.barrier()

        # 确保所有进程都完成了这个数据集的评估
        if args.distributed:
            torch.distributed.barrier()

        metric_logger.synchronize_between_processes()
        # print(f'* {dataset_eval_name} Metric {metric_logger.meters[f"{dataset_eval_name}_Metric"].global_avg:.3f} *', flush=True)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
