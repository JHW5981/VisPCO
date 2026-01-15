import os
import warnings
import logging

# Must set environment variable before importing transformers
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# Filter out warnings related to generation flags
warnings.filterwarnings('ignore', message='.*generation flags.*')
warnings.filterwarnings('ignore', message='.*The following generation flags.*')
warnings.filterwarnings('ignore', message='.*may be ignored.*')
# Set transformers logger level
logging.getLogger('transformers.generation.utils').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import json
import torch
import random
import argparse
import torch.nn as nn
from transformers import AutoProcessor

from eval.eval_engine import evaluate
from dataset.dataset import build_dataset, DataCollatorForSupervisedDataset
from models.qwen2_5vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration_FastV
from utils.utils import (
    init_distributed_mode, seed_everything, 
    get_world_size, get_rank,
    is_main_process, init_weights
)


def get_args_parser():
    parser = argparse.ArgumentParser('Qwen-Fastv-VisPCO evaluation', add_help=False)
    parser.add_argument('--model_name', type=str, default='qwen2.5vl-3b')
    parser.add_argument('--trained_predict_pruning_ratio_path', type=str, default='xxx/workspace/Qwen-Fastv-VisPCO/train_logs/lr_0.0001_epochs_1000_w_1.0_sigma_100.0_alpha_5.0_epsilon_0.005_beta_0.5/predict_pruning_ratio_12.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Dataset Parameters
    parser.add_argument('--valdata_path', default='xxx/workspace/Qwen-Fastv-VisPCO/eval_max_100_dataset.json', type=str, metavar='VALDATA', help="Path to validation data")
    parser.add_argument('--num_workers', default=4, type=int, metavar='NUM_WORKERS', help="Number of workers (default: 16)")
    parser.add_argument('--pin_memory', action='store_true', help="Pin memory (default: False)")
    # Some limitations of Qwen data
    parser.add_argument('--max_pixels', default=4194304, type=int, metavar='MAX_PIXELS', help="Max pixels (default: 28 * 28 * 576)")
    parser.add_argument('--min_pixels', default=400, type=int, metavar='MIN_PIXELS', help="Min pixels (default: 28 * 28 * 16)")
    parser.add_argument('--video_max_frames', default=8, type=int, metavar='VIDEO_MAX_FRAMES', help="Video max frames (default: 8)")
    parser.add_argument('--video_min_frames', default=4, type=int, metavar='VIDEO_MIN_FRAMES', help="Video min frames (default: 4)")
    parser.add_argument('--video_max_pixels', default=1024 * 28 * 28, type=int, metavar='VIDEO_MAX_PIXELS', help="Video max pixels (default: 1024 * 28 * 28)")
    parser.add_argument('--video_min_pixels', default=256 * 28 * 28, type=int, metavar='VIDEO_MIN_PIXELS', help="Video min pixels (default: 256 * 28 * 28)")
    parser.add_argument('--video_fps', default=2, type=float, metavar='VIDEO_FPS', help="Video fps (default: 2)")

    # Distributed Training Parameters
    parser.add_argument('--world_size', default=1, type=int, metavar='WORLD_SIZE', help="Number of nodes for distributed training (default: 1)")
    parser.add_argument('--dist_url', default='env://', type=str, metavar='DIST_URL', help="url used to set up distributed training (default: env://)")

    return parser

def main(args):

    # Configure distributed training
    init_distributed_mode(args)

    # Set random seed
    seed = args.seed + get_rank()
    seed_everything(seed)

    # Model loading
    model_name = args.model_name
    # Precision
    precision_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if model_name == 'qwen2.5vl-7b':
        model_path = "xxx/pretrained_weights/Qwen/Qwen2.5-VL-7B-Instruct"
        config = Qwen2_5_VLConfig.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration_FastV.from_pretrained(
            model_path,
            config=config,
            # ignore_mismatched_sizes=True
        )
        if hasattr(args, 'trained_predict_pruning_ratio_path') and args.trained_predict_pruning_ratio_path:
            state_dict = torch.load(args.trained_predict_pruning_ratio_path, map_location='cpu')
            # Handle prefix issues, such as distributed and module, etc.
            # If all keys have 'module.' prefix, remove it
            new_state_dict = {}
            for k, v in state_dict.items():
                # Handle different prefixes
                new_k = k.replace('module.', 'model.').replace('model.model.language_model.predict_pruning_ratio.', '')
                new_state_dict[new_k] = v
            # Load the filtered state_dict
            missing_keys, unexpected_keys = model.model.language_model.predict_pruning_ratio.load_state_dict(
                new_state_dict, strict=True
            )
            if missing_keys or unexpected_keys:
                print(f"⚠️ There are unmatched items when loading predict_pruning_ratio: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
            print(f"✅ predict_pruning_ratio parameters have been imported from {args.trained_predict_pruning_ratio_path}! ✅")
        else:
            print(f"⚠️ No predict_pruning_ratio parameter path is provided, please check! ⚠️")
        print(f"✅ {model_name} has been successfully loaded! ✅")
    elif model_name == 'qwen2.5vl-3b':
        model_path = "xxx/pretrained_weights/Qwen/Qwen2.5-VL-3B-Instruct"
        config = Qwen2_5_VLConfig.from_pretrained(model_path)
        model = Qwen2_5_VLForConditionalGeneration_FastV.from_pretrained(
            model_path,
            config=config,
            # ignore_mismatched_sizes=True
        )
        if hasattr(args, 'trained_predict_pruning_ratio_path') and args.trained_predict_pruning_ratio_path:
            state_dict = torch.load(args.trained_predict_pruning_ratio_path, map_location='cpu')
            # Handle prefix issues, such as distributed and module, etc.
            # If all keys have 'module.' prefix, remove it
            new_state_dict = {}
            for k, v in state_dict.items():
                # Handle different prefixes
                new_k = k.replace('model.language_model.predict_pruning_ratio.', '')
                new_state_dict[new_k] = v
            # Load the filtered state_dict
            missing_keys, unexpected_keys = model.model.language_model.predict_pruning_ratio.load_state_dict(
                new_state_dict, strict=True
            )
            if missing_keys or unexpected_keys:
                print(f"⚠️ There are unmatched items when loading predict_pruning_ratio: missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")
            print(f"✅ predict_pruning_ratio parameters have been imported from {args.trained_predict_pruning_ratio_path}! ✅")
        else:
            print(f"⚠️ No predict_pruning_ratio parameter path is provided, please check! ⚠️")
        print(f"✅ {model_name} has been successfully loaded! ✅")
    else:
        raise ValueError(f"⚠️ Model {model_name} is currently not supported ⚠️")

    model.eval()
    model = model.to(precision_map[args.precision])
    model = model.to(args.device)

    # Load dataset
    processor = AutoProcessor.from_pretrained(model_path)
    dataset_eval_lists, dataset_eval_names = build_dataset(is_train=False, args=args, processor=processor)

    num_tasks = get_world_size()
    global_rank = get_rank()

    # Choose different samplers depending on whether distributed or not
    if args.distributed:
        sampler_eval_lists = []
        for dataset_eval in dataset_eval_lists:
            sampler_eval_lists.append(torch.utils.data.DistributedSampler(dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed))
    else:
        sampler_eval_lists = None

    eval_dataloader_lists = []
    if args.distributed:
        for dataset_eval, sampler_eval in zip(dataset_eval_lists, sampler_eval_lists):
            eval_dataloader_lists.append(
                torch.utils.data.DataLoader(
                            dataset_eval,
                            sampler=sampler_eval,
                            batch_size=1, # To ensure accuracy, evaluation batch is set to 1
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            drop_last=False,
                            collate_fn=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer),
                )
            )
    else:
        for dataset_eval in dataset_eval_lists:
            eval_dataloader_lists.append(
                torch.utils.data.DataLoader(
                            dataset_eval,
                            batch_size=1, # To ensure accuracy, evaluation batch is set to 1
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            drop_last=False,
                            shuffle=False,
                            collate_fn=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer),
                )
            )
    for ratio in [0.5]:
        test_stats = evaluate(eval_dataloader_lists, dataset_eval_names, model, processor, args, ratio=ratio)
        flops = test_stats.pop('flops')
        max_budgets = test_stats.pop('max_budgets')
        min_budgets = test_stats.pop('min_budgets')
        all_acc = sum(test_stats.values())
        if is_main_process():
            log_stats = {
                'ratio': ratio,
                'flops': round(flops, 3),
                'max_budgets': round(max_budgets, 3),
                'min_budgets': round(min_budgets, 3),
                'all_acc': round(all_acc, 2),
                **{f'test_{k}': round(v, 2) for k, v in test_stats.items()},
                }
            with open(os.path.join("xxx/workspace/Qwen-Fastv-VisPCO/results", "eval_ours.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if args.distributed:
            torch.distributed.barrier()

    print(f"✅ {model_name} evaluation finished! ✅")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
