import json
import torch
from torch import optim as optim
import math
import numpy as np
from torch import inf

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
# from timm.optim.nadam import Nadam
# from timm.optim.novograd import NovoGrad
from timm.optim.nvnovograd import NvNovoGrad
# from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

def get_parameter_groups(model, weight_decay, skip_list=[], bone_lr_scale=0.01, fix_step=1):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif 'predict_pruning_ratio' in name:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = 'new_param_no_decay'
                this_weight_decay = 0
            else:
                group_name = 'new_param'
                this_weight_decay = weight_decay
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        
        if group_name not in parameter_group_names:  
            if group_name == 'decay':
                scale = bone_lr_scale
                fix_step = fix_step
            elif group_name == 'no_decay':
                scale = bone_lr_scale
                fix_step = fix_step
            else:
                scale = 1.
                fix_step = 0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "fix_step": fix_step
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "fix_step": fix_step
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, skip_list=[], bone_lr_scale=0.01, fix_step=1):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    parameters = get_parameter_groups(model, weight_decay, skip_list, bone_lr_scale, fix_step)
    weight_decay = 0.

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    # elif opt_lower == 'nadam':
    #     optimizer = Nadam(parameters, **opt_args)
    # elif opt_lower == 'radam':
    #     optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'novograd':
    #     optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = 0
    total_iters = int(epochs * niter_per_ep)
    if isinstance(warmup_steps, int) and warmup_steps > 0:
        warmup_iters = warmup_steps
    elif isinstance(warmup_steps, float) and warmup_steps > 0:
        warmup_iters = int(total_iters * warmup_steps)
    print("Set warmup steps = ", warmup_iters)
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

class NativeScalerWithGradNormCount:
    state_dict_key = "scaler"
    
    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)
    
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, 
                 create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
        if update_grad:
            # unscale_ 在 enabled=False 时是安全的,不会报错
            self._scaler.unscale_(optimizer)
            
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
