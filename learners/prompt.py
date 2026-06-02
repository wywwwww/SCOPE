from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule, CosineSchedulerIter
from torch.autograd import Variable, Function
from collections import defaultdict
import time
import torch
class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)


        self.det_opt = {}  # layer -> optimizer



    def _init_det_optimizers_for_task(self, task_id, layers=None, lr=1e-3):
        scope = self.model.prompt
        self.det_opt = {}
        if layers is None:
            layers = scope.e_layers
        for l in sorted(layers):
            scope.init_detector(l, task_id)
            self.det_opt[l] = torch.optim.Adam(scope.detectors[scope._det_key(l, task_id)].parameters(), lr=lr)

    def update_model(self, inputs, targets):

        # logits
        logits, prompt_loss = self.model(inputs, targets, train=True, cls_mean=self.cls_mean, optimizer=self.optimizer) # logits=cls_token if pen=True, else self.model.last(cls_token)
        logits = logits[:,:self.valid_out_dim]

        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf') # TODO: this gives inf loss if self.memory_size > 0
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        if self.epoch == 0:
            # ===== detector training (detach) =====
            scope = self.model.prompt
            t = self.task_count
            for l, h in scope.cached_query.items():
                if l in self.det_opt:
                    scope.detector_step(l, t, h, self.det_opt[l])

        return total_loss.detach(), logits

    def freeze_param(self, param, optimizer):
        param.requires_grad = False
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p is not param]
        optimizer.state.pop(param, None)

    def _flush_rho_window(self, progress=1.0):
        scope = self.model.prompt

        if not hasattr(scope, "rho_window_sum"):
            scope.rho_window_sum = defaultdict(float)
        if not hasattr(scope, "rho_window_cnt"):
            scope.rho_window_cnt = defaultdict(int)
        if not hasattr(scope, "rho_window_progress"):
            scope.rho_window_progress = 0.0
        if not hasattr(scope, "rho_window_epochs"):
            scope.rho_window_epochs = 1.0

        expanded_layers = {l for l, v in scope.cold_started.items() if v}


        for l in list(expanded_layers):
            if scope.rho_cnt.get(l, 0) == 0:
                continue

            scope.rho_window_sum[l] += scope.rho_sum[l]
            scope.rho_window_cnt[l] += scope.rho_cnt[l]


            scope.rho_sum[l] = 0.0
            scope.rho_cnt[l] = 0


        scope.rho_window_progress += progress


        if scope.rho_window_progress < scope.rho_window_epochs:
            return []


        candidates = []
        for l in list(expanded_layers):
            if scope.rho_window_cnt.get(l, 0) == 0:
                continue

            rho = scope.rho_window_sum[l] / max(1, scope.rho_window_cnt[l])
            scope.rho_hist[l].append(rho)


            scope.rho_window_sum[l] = 0.0
            scope.rho_window_cnt[l] = 0


            if self.epoch - scope.last_expand_epoch[l] < scope.min_wait_epochs_after_expand:
                continue

            hist = scope.rho_hist[l]
            print(f"[Layer {l}] rho_hist = {hist}")

            if len(hist) < 2:
                continue

            if hist[-1] >= hist[-2]:
                scope.no_improve_count[l] += 1
            else:
                scope.no_improve_count[l] = 0

            trigger = (scope.no_improve_count[l] >= 2) if scope.need_two_strikes else (scope.no_improve_count[l] >= 1)
            if trigger:
                candidates.append(l)

        # 5) reset window progress
        scope.rho_window_progress = 0.0
        return candidates

    def end_epoch(self):
        scope = self.model.prompt

        progress = 0.5 if getattr(scope, "rho_window_epochs", 1.0) == 0.5 else 1.0
        candidates = self._flush_rho_window(progress=progress)

        if len(candidates) > 0:
            next_l = min(candidates)


            if (hasattr(scope, "epoch_cached_query_det") and next_l in scope.epoch_cached_query_det
                    and len(scope.epoch_cached_query_det[next_l]) > 0):
                h = torch.cat(scope.epoch_cached_query_det[next_l], dim=0).to(next(self.model.parameters()).device)
                with torch.no_grad():
                    mu = scope.compute_residual_init_key(next_l, h, q=0.95)
                    scope.residual_init_key[next_l] = mu

            scope.expand_now_layers.add(next_l)
            scope.expand_layers.add(next_l)
            scope.last_expand_epoch[next_l] = self.epoch
            scope.no_improve_count[next_l] = 0

            print(f"[Epoch {self.epoch}] rho plateau -> expand another prompt at layer {next_l}")
            return

        if not hasattr(scope, "pending_layers"):
            return

        new_expand = []
        done_layers = []
        q = 0.95
        for l in list(scope.pending_layers):
            if l not in scope.epoch_cached_query_det or len(scope.epoch_cached_query_det[l]) == 0:
                continue
            h = torch.cat(scope.epoch_cached_query_det[l], dim=0)

            ok = False
            for t_prev in range(self.task_count):
                stats = scope.detector_stats.get((l, t_prev), None)
                if stats is None:
                    continue

                err_new = scope.detector_score(l, t_prev, h)
                err_new_q = torch.quantile(err_new, q)
                errs_old = stats["errs"]  # Tensor [N]
                err_old_q = torch.quantile(errs_old, q)

                if err_new_q <= err_old_q:
                    ok = True
                    break

            if ok:
                done_layers.append(l)
            else:
                new_expand.append(l)


        for l in done_layers:
            scope.pending_layers.remove(l)


        if new_expand:
            next_l = min(new_expand)
            h = torch.cat(scope.epoch_cached_query_det[next_l], dim=0)
            with torch.no_grad():
                mu = scope.compute_residual_init_key(next_l, h, q=0.95)
                scope.residual_init_key[next_l] = mu

            scope.expand_now_layers.add(next_l)
            scope.expand_layers.add(next_l)
            scope.last_expand_epoch[next_l] = self.epoch
            scope.pending_layers.remove(next_l)
            scope.no_improve_count[next_l] = 0
            l = next_l
            print(f"[Epoch {self.epoch}] detector says still shift -> expand new layer {next_l}")

        scope.epoch_cached_query_det.clear()

    @torch.no_grad()
    def begin_task(self, train_loader, num_probe_batches=15):
        scope = self.model.prompt
        cur_t = self.task_count
        scope.epoch_cached_query_det = defaultdict(list)
        scope.expand_flag = False

        # ===== rho tracking =====
        scope.rho_sum = defaultdict(float)
        scope.rho_cnt = defaultdict(int)
        scope.rho_hist = defaultdict(list)

        scope.rho_window_sum = defaultdict(float)
        scope.rho_window_cnt = defaultdict(int)
        scope.rho_window_progress = 0.0
        scope.rho_window_epochs = 1.0

        scope.expand_now_layers = set()
        scope.last_expand_epoch = defaultdict(lambda: -999)
        scope.no_improve_count = defaultdict(int)
        scope.min_wait_epochs_after_expand = 1
        scope.need_two_strikes = True

        self.det_opt = {}
        scope.new_detector_layers = set()
        scope.cur_task_id = cur_t
        scope.cold_started = defaultdict(bool)


        if cur_t == 0:
            scope.new_detector_layers = set(scope.e_layers)
            self._init_det_optimizers_for_task(cur_t, layers=scope.new_detector_layers, lr=1e-4)
            scope.expand_layers = set(scope.e_layers)
            return
        torch.cuda.synchronize()
        start = time.time()


        h_by_layer_sum = {l: [] for l in scope.e_layers}
        it = iter(train_loader)
        for _ in range(num_probe_batches):
            try:
                x, y, task = next(it)
            except StopIteration:
                break
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
            saved = scope.expand_layers
            scope.expand_layers = set()

            _ = self.model(x, y, train=True, cls_mean=self.cls_mean, optimizer=self.optimizer)
            scope.expand_layers = saved

            for l in scope.e_layers:
                if l in scope.cached_query:
                    h_by_layer_sum[l].append(scope.cached_query[l])

        h_by_layer = {l: torch.cat(h_by_layer_sum[l], dim=0) for l in scope.e_layers if len(h_by_layer_sum[l]) > 0}


        shift_stats = {}
        expand = []
        q = 0.95
        for l, h in h_by_layer.items():
            ok = False
            shift_stats[l] = []
            for t_prev in range(cur_t):
                if (l, t_prev) not in scope.detector_stats:
                    continue

                errs_old = scope.detector_stats[(l, t_prev)]["errs"]  # Tensor [N]
                thr_q = torch.quantile(errs_old, q).item()


                errs_new = scope.detector_score(l, t_prev, h)  # Tensor [M]
                err_new_q = torch.quantile(errs_new, q).item()


                shift = err_new_q - thr_q
                shift_pos = max(shift, 0.0)
                shift_stats[l].append(shift_pos)


                if err_new_q <= thr_q:
                    ok = True
                    break
            if not ok:
                expand.append(l)

        # ===== compute mean shift per layer =====
        layer_mean_shift = {}

        for l, shifts in shift_stats.items():
            if len(shifts) > 0:
                layer_mean_shift[l] = sum(shifts) / len(shifts)

        # ===== compute mean shift over expanded layers =====
        expanded_shifts = []

        for l in expand:
            if l in layer_mean_shift:
                expanded_shifts.append(layer_mean_shift[l])

        if len(expanded_shifts) > 0:
            mean_shift_expanded = sum(expanded_shifts) / len(expanded_shifts)
        else:
            mean_shift_expanded = 0.0

        print(f"[Task {cur_t}] mean shift (expanded layers): {mean_shift_expanded:.4e}")
        torch.cuda.synchronize()

        scope.new_detector_layers = set(expand)
        if len(scope.new_detector_layers) > 0:
            self._init_det_optimizers_for_task(cur_t, layers=scope.new_detector_layers, lr=1e-4)

        if len(expand) > 0:
            scope.expand_flag = True
            shallow = min(expand)
            scope.expand_layers = {shallow}
            scope.pending_layers = sorted([l for l in expand if l != shallow])
        else:
            scope.expand_layers = set()
            scope.pending_layers = []
        scope.task_has_expansion = (len(scope.expand_layers) > 0)
        if scope.task_has_expansion:
            for l in scope.expand_layers:
            # for l in scope.e_layers:
                for k in getattr(scope, f"e_k_{l}"):
                    k.requires_grad = False
                for p in getattr(scope, f"e_p_{l}"):
                    p.requires_grad = False
        else:
            for l in scope.e_layers:
                for k in getattr(scope, f"e_k_{l}"):
                    k.requires_grad = False
                for p in getattr(scope, f"e_p_{l}"):
                    p.requires_grad = True

        self.log(f"[Begin Task {cur_t}] expand_layers = {sorted(list(scope.expand_layers))}")
        # ===== warmup init for this task =====
        scope.new_prompt_indices = defaultdict(list)
        scope.cur_epoch_in_task = 0


        scope.enable_warmup_bias = scope.task_has_expansion
        scope.residual_init_key = {}

        if scope.task_has_expansion:
            with torch.no_grad():
                for l in scope.expand_layers:
                    if l not in h_by_layer:
                        continue
                    h = h_by_layer[l]  # [M, d] from probe cached_query


                    K_list = getattr(scope, f"e_k_{l}")
                    if len(K_list) == 0:

                        mu = F.normalize(h.mean(dim=0, keepdim=True), dim=1)
                        scope.residual_init_key[l] = mu
                        continue

                    K_old = torch.cat(list(K_list), dim=0)  # [P_old, d]
                    K_old = F.normalize(K_old, dim=1)
                    h_norm = F.normalize(h, dim=1)

                    # residual = 1 - max cosine(h, K_old)
                    cos = torch.einsum("md,pd->mp", h_norm, K_old)  # [M, P_old]
                    max_cos = cos.max(dim=1).values  # [M]
                    residual = 1.0 - max_cos  # [M]


                    q = 0.95
                    thr = torch.quantile(residual, q).item()
                    idx = (residual >= thr).nonzero(as_tuple=False).squeeze(1)

                    if idx.numel() == 0:

                        mu = F.normalize(h_norm.mean(dim=0, keepdim=True), dim=1)
                    else:
                        mu = F.normalize(h_norm[idx].mean(dim=0, keepdim=True), dim=1)

                    scope.residual_init_key[l] = mu  # [1, d]

    @torch.no_grad()
    def end_task(self, train_loader, num_stat_batches=15):
        scope = self.model.prompt
        cur_t = self.task_count
        new_detector_layers = set(getattr(scope, "new_detector_layers", set()))
        err_buf = {l: [] for l in new_detector_layers}

        if len(new_detector_layers) == 0:
            self.log(f"[End Task {cur_t}] no new detectors needed.")
            return

        it = iter(train_loader)
        for _ in range(num_stat_batches):
            try:
                x, y, task = next(it)
            except StopIteration:
                break
            if self.gpu:
                x = x.cuda()
                y = y.cuda()
            saved_expand_layers = scope.expand_layers
            scope.expand_layers = set()
            _ = self.model(x, y, train=True, cls_mean=self.cls_mean, optimizer=self.optimizer)
            scope.expand_layers = saved_expand_layers
            for l in new_detector_layers:
                if l in scope.cached_query:
                    h = scope.cached_query[l]
                    err = scope.detector_score(l, cur_t, h).detach().cpu()
                    err_buf[l].append(err)

        for l in new_detector_layers:
            if len(err_buf[l]) > 0:
                errs = torch.cat(err_buf[l], dim=0)
                scope.finalize_detector_threshold(l, cur_t, errs, k_sigma=1.0)
                scope.freeze_detector(l, cur_t)

        self.log(f"[End Task {cur_t}] new detectors frozen & thresholds saved: {sorted(new_detector_layers)}")

    # sets model optimizers
    def init_optimizer(self):

        if len(self.config['gpuid']) > 1:
            base_params = list(self.model.module.prompt.parameters())
            base_fc_params = list(self.model.module.last.parameters())
        else:
            base_params = list(self.model.prompt.parameters())
            base_fc_params = list(self.model.last.parameters())
        base_params = {'params': base_params, 'lr': self.config['lr']*5, 'weight_decay': self.config['weight_decay']} # HiDe-Prompt - larger_prompt_lr
        base_fc_params = {'params': base_fc_params, 'lr': self.config['lr'], 'weight_decay': self.config['weight_decay']} 
        optimizer_arg = [base_params, base_fc_params]

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](optimizer_arg)
        
        # create schedules 
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)
        elif self.schedule_type == 'coswm':
            # print(self.config)
            scheduler_cfg = {
                'base_value': [self.config['lr']*5, self.config['lr']], 
                'final_value': [1e-6, 1e-6], 
                'optimizer': self.optimizer, 
                'iter_step': self.config['iter_step'], 
                'n_epochs': self.config['schedule'][-1], 
                'last_epoch': -1, 
                'warmup_epochs': self.config['schedule'][1], 
                'start_warmup_value': 0, 
                'freeze_iters': self.config['schedule'][0]
            }
            self.scheduler = CosineSchedulerIter(**scheduler_cfg)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# Our method
class SCOPE(Prompt):

    def __init__(self, learner_config):
        super(SCOPE, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, 
                                                                               prompt_flag = 'scope',
                                                                               prompt_param=self.prompt_param,
                                                                               pretrained=cfg['pretrained_weight']) # vit_pt_imnet
        return model

# @inproceedings{smith2023coda,
#   title={CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning},
#   author={Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={11909--11919},
#   year={2023}
# }
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model
