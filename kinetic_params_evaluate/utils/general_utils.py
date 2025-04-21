# import numpy as np
# import torch.utils.data as Data
# import torch
# from torch.optim import Optimizer
# import math
# from typing import Callable, Iterable, Tuple
# from torch.distributions.bernoulli import Bernoulli
#
#
#
# class MTLKineticParamDataset(Data.Dataset):
#     def __init__(self, df, input_features, input_idx):
#         self.df = df
#         self.input_features = input_features
#         self.input_idx = input_idx
#
#     def __len__(self):
#         return len(self.input_idx)
#
#     def __getitem__(self, idx):
#         idx_in_df = self.input_idx[idx]
#         df_idx = self.df.loc[idx_in_df]
#
#         if self.input_features[-1] == 'logkcatkm':
#             label_km_values = np.array(df_idx[self.input_features[-3]])
#             label_kcat_values = np.array(df_idx[self.input_features[-2]])
#             label_kcatkm_values = np.array(df_idx[self.input_features[-1]])
#
#             input_features_columns = self.input_features[:-3]  # 前几列作为输入特征
#             concatenated_series = [np.array(item) if isinstance(item, list) else np.array([item]) for item in df_idx[input_features_columns]]  # 提取这些列的值
#             input_values = np.concatenate(concatenated_series)  # 拼接这些列的值
#             return input_values, label_km_values, label_kcat_values, label_kcatkm_values
#
#         else:
#             label_km_values = np.array(df_idx[self.input_features[-2]])
#             label_kcat_values = np.array(df_idx[self.input_features[-1]])
#
#             input_features_columns = self.input_features[:-2]  # 前几列作为输入特征
#
#             concatenated_series = [np.array(item) if isinstance(item, list) else np.array([item]) for item in df_idx[input_features_columns]]  # 提取这些列的值
#             input_values = np.concatenate(concatenated_series)  # 拼接这些列的值
#
#             return input_values, label_km_values, label_kcat_values
#
#
# class WarmCosine:
#     def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
#         if warmup is None:
#             self.warmup = 0
#         else:
#             warmup_step = int(warmup)
#             assert warmup_step > 0
#             self.warmup = warmup_step
#             self.lr_step = (1 - eta_min) / warmup_step
#         self.tmax = int(tmax)
#         self.eta_min = eta_min
#
#     def step(self, step):
#         if step >= self.warmup:
#             return (
#                     self.eta_min
#                     + (1 - self.eta_min)
#                     * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
#                     / 2
#             )
#
#         else:
#             return self.eta_min + self.lr_step * step
#
#
# class ChildTuningAdamW(Optimizer):
#     def __init__(
#             self,
#             params: Iterable[torch.nn.parameter.Parameter],
#             lr: float = 1e-3,
#             betas: Tuple[float, float] = (0.9, 0.999),
#             eps: float = 1e-6,
#             weight_decay: float = 0.0,
#             correct_bias: bool = True,
#             reserve_p=1.0,
#             mode=None
#     ):
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
#         super().__init__(params, defaults)
#
#         self.gradient_mask = None
#         self.reserve_p = reserve_p
#         self.mode = mode
#
#     def set_gradient_mask(self, gradient_mask):
#         self.gradient_mask = gradient_mask
#
#     def step(self, closure: Callable = None):
#         """
#         Performs a single optimization step.
#
#         Arguments:
#             closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
#
#                 # =================== HACK BEGIN =======================
#                 if self.mode is not None:
#                     if self.mode == 'ChildTuning-D':
#                         if p in self.gradient_mask:
#                             grad *= self.gradient_mask[p]
#                     else:
#                         # ChildTuning-F
#                         grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
#                         grad *= grad_mask.sample() / self.reserve_p
#                 # =================== HACK END =======================
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(p.data)
#
#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]
#
#                 state["step"] += 1
#
#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])
#
#                 step_size = group["lr"]
#                 if group["correct_bias"]:  # No bias correction for Bert
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
#
#                 p.data.addcdiv_(exp_avg, denom, value=-step_size)
#
#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 # Add weight decay at the end (fixed version)
#                 p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
#
#         return loss
#
#
# def mtl_return_loader(df_input, input_features, input_idx, batch_size, shuffle=True, seed=0):
#     dataset = MTLKineticParamDataset(df_input, input_features, input_idx)
#
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     g = torch.Generator()
#
#     # 构造 DataLoader
#     # dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g, pin_memory=True, num_workers=8)
#     dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
#
#     return dataloader
#
#
#
