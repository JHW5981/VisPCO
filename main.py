import argparse
import os
import sys
import logging
import time
import json
import torch
import datetime
import copy

from transformers import AutoProcessor
from utils.logger import MyLogger, MetricLogger, SmoothedValue
from utils.utils import (
    init_distributed_mode, seed_everything, 
    get_world_size, get_rank,
    is_main_process, init_weights
)
from dataset.dataset import build_dataset, DataCollatorForSupervisedDataset
from models.qwen2_5vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration_FastV, Qwen2_5_VLConfig
from train.optim_factory import create_optimizer, cosine_scheduler
from train.engine import train_one_epoch, evaluate
from train.losses import create_criterion
from tqdm import tqdm
from thop import profile

# from peft import get_peft_model
# from peft import LoraConfig, TaskType

torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser("Dynamic-VLM training scripts")
    
    # Model Parameters
    parser.add_argument("--model_name", default='qwen2.5vl-3b', type=str, metavar='MODEL', help="Name of model to use")
    parser.add_argument("--resource_model_path", default='/mnt/inaisfs/home/test3/jihuawei/workspace/Qwen-Fastv-VisPCO/resource_network_weights/macs_model.pth', type=str, metavar='RESOURCE_MODEL_PATH', help="Path to resource model")

    # Dataset Parameters
    parser.add_argument('--trdata_path', default='/mnt/inaisfs/home/test3/jihuawei/workspace/Qwen-Fastv-VisPCO/train_10_uniform_area_single_image.json', type=str, metavar='TRDATA', help="Path to training data")
    parser.add_argument('--valdata_path', default='/mnt/inaisfs/home/test3/jihuawei/workspace/Qwen-Fastv-VisPCO/eval_max_5_dataset.json', type=str, metavar='VALDATA', help="Path to validation data")
    parser.add_argument('--batch_size', default=2, type=int, metavar='BATCH_SIZE', help="Batch size (default: 16)")
    parser.add_argument('--num_workers', default=4, type=int, metavar='NUM_WORKERS', help="Number of workers (default: 16)")
    parser.add_argument('--pin_memory', action='store_true', help="Pin memory (default: False)")
    parser.add_argument('--seed', default=42, type=int, metavar='SEED', help="Random seed (default: 42)")
    # Qwen数据的一些限制
    parser.add_argument('--max_pixels', default=4194304, type=int, metavar='MAX_PIXELS', help="Max pixels (default: 28 * 28 * 576)")
    parser.add_argument('--min_pixels', default=400, type=int, metavar='MIN_PIXELS', help="Min pixels (default: 28 * 28 * 16)")
    parser.add_argument('--video_max_frames', default=8, type=int, metavar='VIDEO_MAX_FRAMES', help="Video max frames (default: 8)")
    parser.add_argument('--video_min_frames', default=4, type=int, metavar='VIDEO_MIN_FRAMES', help="Video min frames (default: 4)")
    parser.add_argument('--video_max_pixels', default=1024 * 28 * 28, type=int, metavar='VIDEO_MAX_PIXELS', help="Video max pixels (default: 1024 * 28 * 28)")
    parser.add_argument('--video_min_pixels', default=256 * 28 * 28, type=int, metavar='VIDEO_MIN_PIXELS', help="Video min pixels (default: 256 * 28 * 28)")
    parser.add_argument('--video_fps', default=2, type=float, metavar='VIDEO_FPS', help="Video fps (default: 2)")
    
    # Optimizer Parameters
    parser.add_argument("--opt", default='adamw', type=str, metavar='OPTIMIZER', help="Optimizer (default: adamw_torch)")
    parser.add_argument("--lr", default=4e-4, type=float, metavar='LR', help="Learning rate (default: 4e-3)")
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--warmup_steps', default=0.01, type=float, metavar='N',
                        help='num of steps to warmup LR')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_scale', type=float, default=0.01)
    parser.add_argument('--fix_step', type=int, default=1, metavar='FIX_STEP',
                        help='Fix step (default: None, no fixing)')
    parser.add_argument('--max_norm', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    # 拉格朗日乘子法的训练参数
    parser.add_argument("--w", default=1, type=float)
    parser.add_argument("--sigma", default=100, type=float)
    parser.add_argument("--alpha", default=2, type=float)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--beta", default=0.5, type=float)

    # Training Parameters
    parser.add_argument('--epochs', default=3, type=int, metavar='EPOCHS', help="Number of epochs (default: 10)")
    parser.add_argument('--device', default='cuda', type=str, metavar='DEVICE', help="Device (default: cuda)")
    parser.add_argument('--precision', default='bf16', type=str, metavar='PRECISION', help="Precision (default: fp32)")
    parser.add_argument('--update_freq', default=1, type=int, metavar='UPDATE_FREQ', help="Update frequency (default: 1)")
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--log_dir', default='/mnt/inaisfs/home/test3/jihuawei/workspace/Qwen-Fastv-VisPCO/train_logs/debug/', type=str, metavar='LOG_DIR', help="Path to save logs (default: logs)")
    parser.add_argument('--log_file', default='log.txt', type=str)
    parser.add_argument('--save_ckpt_path', default='/mnt/inaisfs/home/test3/jihuawei/workspace/Qwen-Fastv-VisPCO/outputs', type=str, metavar='SAVE_CKPT_PATH', help="Path to save checkpoint, empty for no saving (default: output)")
    parser.add_argument('--save_ckpt_freq', default=1, type=int, metavar='SAVE_CKPT_FREQ', help="Save checkpoint frequency (default: 1)")
    parser.add_argument('--save_ckpt_num', default=3, type=int, metavar='SAVE_CKPT_NUM', help="Number of checkpoints to save (default: 3)")
    parser.add_argument('--finetuning', action='store_true', help="Finetuning (default: False)")

    # Evaluation Parameters
    parser.add_argument('--eval', default=1, type=int, help="隔几次eval一次")
    parser.add_argument('--disable_eval', action='store_true', 
                        help="Disable evaluation during training (default: False)")

    # Distributed Training Parameters
    parser.add_argument('--world_size', default=1, type=int, metavar='WORLD_SIZE', help="Number of nodes for distributed training (default: 1)")
    parser.add_argument('--dist_url', default='env://', type=str, metavar='DIST_URL', help="url used to set up distributed training (default: env://)")

    # Dynamic-VLM Parameters
    parser.add_argument('--throughput', action='store_true', help="See throughput during training (default: False)")
    parser.add_argument('--print_mode', action='store_false', help="Finetuning (default: True)")
    return parser

def main(args):
    # 配置分布式训练
    init_distributed_mode(args)

    # 配置随机种子
    seed = args.seed + get_rank()
    seed_everything(seed)

    # 模型加载
    model_name = args.model_name
    # 精度
    precision_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if model_name == 'qwen2.5vl-7b':
        model_path = "/mnt/inaisfs/home/test3/jihuawei/pretrained_weights/Qwen/Qwen2.5-VL-7B-Instruct"
        config = Qwen2_5_VLConfig.from_pretrained(model_path)
        config.finetuning = args.finetuning
        model = Qwen2_5_VLForConditionalGeneration_FastV.from_pretrained(
            model_path,
            config=config,
            # ignore_mismatched_sizes=True
        )
        teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            # ignore_mismatched_sizes=True
        )    

        # 单独对student模型的predict_pruning_ratio进行初始化，不然就跑飞了
        model.model.language_model.predict_pruning_ratio.apply(init_weights)
        print(f"✅ {model_name}已经导入完成！✅")
    elif model_name == 'qwen2.5vl-3b':
        model_path = "/mnt/inaisfs/home/test3/jihuawei/pretrained_weights/Qwen/Qwen2.5-VL-3B-Instruct"
        config = Qwen2_5_VLConfig.from_pretrained(model_path)
        config.finetuning = args.finetuning
        model = Qwen2_5_VLForConditionalGeneration_FastV.from_pretrained(
            model_path,
            config=config,
            # ignore_mismatched_sizes=True
        )
        teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            # ignore_mismatched_sizes=True
        )    

        # 单独对student模型的predict_pruning_ratio进行初始化，不然就跑飞了
        model.model.language_model.predict_pruning_ratio.apply(init_weights)
        print(f"✅ {model_name}已经导入完成！✅")
    else:
        raise ValueError(f"⚠️ 模型 {model_name} 暂不支持 ⚠️")

    model.eval()
    teacher_model.eval()
    model = model.to(precision_map[args.precision])
    model = model.to(args.device)
    teacher_model = teacher_model.to(precision_map[args.precision])
    teacher_model = teacher_model.to(args.device)

    # 再次确保所有参数都在正确的设备上
    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            print(f"Warning: {name} is still on CPU!")
            param.data = param.data.to(args.device)

    for name, buf in model.named_buffers():
        if buf.device.type == 'cpu':
            print(f"Warning: buffer {name} is still on CPU!")
            buf.data = buf.data.to(args.device)

    # 解冻predictor
    for name, param in model.named_parameters():
        if 'predict_pruning_ratio' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    print(f'Number of Trainable params:{n_parameters:.1f}B')

    # 冻结teacher model的所有参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 数据集加载
    processor = AutoProcessor.from_pretrained(model_path)
    dataset_train, _ = build_dataset(is_train=True, args=args, processor=processor)
    dataset_eval_lists, dataset_eval_names = build_dataset(is_train=False, args=args, processor=processor)

    num_tasks = get_world_size()
    global_rank = get_rank()

    # 根据是否分布式选择不同的采样器
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
        if dataset_eval_lists is not None:
            sampler_eval_lists = []
            for dataset_eval in dataset_eval_lists:
                sampler_eval_lists.append(torch.utils.data.DistributedSampler(dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed))
    else:
        sampler_train = None
        sampler_eval_lists = None

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        shuffle=(sampler_train is None),  # 只在非分布式时使用shuffle
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer),
    )
    eval_dataloader_lists = []
    if args.distributed:
        for dataset_eval, sampler_eval in zip(dataset_eval_lists, sampler_eval_lists):
            eval_dataloader_lists.append(
                torch.utils.data.DataLoader(
                            dataset_eval,
                            sampler=sampler_eval,
                            batch_size=1, # 为了保证准确evaluation的batch用1
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
                            batch_size=1, # 为了保证准确evaluation的batch用1
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            drop_last=False,
                            shuffle=False,
                            collate_fn=DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer),
                )
            )

    # 日志记录
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        logger_writer = MyLogger(log_dir=args.log_dir)
    else:
        logger_writer = None

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # 训练参数
    total_batch_size = args.batch_size * args.update_freq * get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)
    
    # 创建优化器
    optimizer = create_optimizer(args, model, skip_list=[], bone_lr_scale=args.lr_scale, fix_step=args.fix_step)
    # 学习率调度器
    lr_schedule_values = cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_steps=args.warmup_steps)
    # 创建损失函数
    criterion = create_criterion(teacher_model, model, args)

    # 第一次测试，并且用于初始化拉格朗日乘子
    sub_flops = []
    test_stats = evaluate(eval_dataloader_lists, dataset_eval_names, model, processor, args)
    flop_loss = test_stats.pop('flop_loss')
    sub_flops.append(flop_loss)
    flops = test_stats.pop('flops')
    max_budgets = test_stats.pop('max_budgets')
    min_budgets = test_stats.pop('min_budgets')
    all_acc = sum(test_stats.values())
    log_stats = {
        'flop_loss': round(flop_loss, 3),
        'flops': round(flops, 3),
        'max_budgets': round(max_budgets, 3),
        'min_budgets': round(min_budgets, 3),
        'all_acc': round(all_acc, 2),
        **{f'{k}': round(v, 2) for k, v in test_stats.items()},
    }
    print("=" * 50)
    print("🌟 Evaluation Log Stats 🌟")
    for k, v in log_stats.items():
        print(f"{k:>16} : {v}")
    print("=" * 50)

    print(f"初始的w: {args.w}, 初始的sigma: {args.sigma}")

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        if logger_writer is not None:
            logger_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            train_dataloader, model, optimizer, criterion,
            epoch, metric_logger=metric_logger, logger_writer=logger_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, num_training_steps_per_epoch=num_training_steps_per_epoch, args=args
        )

        train_log_stats = {**{f'train_{k}': round(v, 2) for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        # 用测试集上的loss代替
        test_stats = evaluate(eval_dataloader_lists, dataset_eval_names, model, processor, args)
        flop_loss = test_stats.pop('flop_loss')
        sub_flops.append(flop_loss)
        flops = test_stats.pop('flops')
        max_budgets = test_stats.pop('max_budgets')
        min_budgets = test_stats.pop('min_budgets')
        all_acc = sum(test_stats.values())
        log_stats = {
            'flop_loss': round(flop_loss, 3),
            'flops': round(flops, 3),
            'max_budgets': round(max_budgets, 3),
            'min_budgets': round(min_budgets, 3),
            'all_acc': round(all_acc, 2),
            **{f'{k}': round(v, 2) for k, v in test_stats.items()},
        }
        print("=" * 50)
        print("🌟 Evaluation Log Stats 🌟")
        for k, v in log_stats.items():
            print(f"{k:>16} : {v}")
        print("=" * 50)

        # 更新拉格朗日乘子
        # if abs(sub_flops[-1]) < args.epsilon:
        if len(sub_flops) > 4 and abs(sub_flops[-1]) < args.epsilon and abs(sub_flops[-2]) < args.epsilon and abs(sub_flops[-3]) < args.epsilon:
            if is_main_process():
                saved_checkpoint_state_dict = {}
                for name, value in model.state_dict().items():
                    if 'predict_pruning_ratio' in name:
                        saved_checkpoint_state_dict[name] = value
                torch.save(saved_checkpoint_state_dict, os.path.join(args.log_dir, f"predict_pruning_ratio_{epoch}.pth"))    
            break
        
        if abs(sub_flops[-1]) / abs(sub_flops[-2]) >= args.beta:
            args.sigma = args.sigma * args.alpha
        args.w = args.w - args.sigma * (sub_flops[-1])

        if args.distributed:
            # 同步args.w和args.sigma到各个进程（取主进程的为准）
            w_tensor = torch.tensor([args.w], dtype=torch.float32, device=args.device)
            sigma_tensor = torch.tensor([args.sigma], dtype=torch.float32, device=args.device)
            torch.distributed.broadcast(w_tensor, src=0)
            torch.distributed.broadcast(sigma_tensor, src=0)
            args.w = w_tensor.item()
            args.sigma = sigma_tensor.item()
        # 更新criterion
        criterion.w = args.w
        criterion.sigma = args.sigma
        print(f"更新后的w: {args.w}, 更新后的sigma: {args.sigma}")
        

        if args.log_dir is not None and is_main_process():
            if logger_writer is not None:
                logger_writer.flush()
            with open(os.path.join(args.log_dir, args.log_file), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(train_log_stats) + "\n")

        if is_main_process():
            saved_checkpoint_state_dict = {}
            for name, value in model.state_dict().items():
                if 'predict_pruning_ratio' in name:
                    saved_checkpoint_state_dict[name] = value
            torch.save(saved_checkpoint_state_dict, os.path.join(args.log_dir, f"predict_pruning_ratio_{epoch}.pth"))    
        if args.distributed:
            torch.distributed.barrier()

    if args.distributed:
        torch.distributed.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    args.log_dir = args.log_dir + 'lr_' + str(args.lr) + '_epochs_' + str(args.epochs) + '_w_' + str(args.w) + '_sigma_' + str(args.sigma) + '_alpha_' + str(args.alpha) + '_epsilon_' + str(args.epsilon) + '_beta_' + str(args.beta)
    log_file_path = f"{args.log_dir}/main_{args.log_dir.split('/')[-1]}.out"
    os.makedirs(f"{args.log_dir}", exist_ok=True)
    sys.stdout = open(log_file_path, "a")
    sys.stderr = sys.stdout

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    main(args)
