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
import copy
from utils.schedulers import CosineSchedule
import time
import torch
class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.model = self.create_model()
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()

        # storing class mean and covariance
        # self.cls_mean = dict()
        # self.cls_cov = dict() # not work for DataParallel
        self.cls_mean = dict() # for mapped targets, which are always ordered
        self.cls_cov = dict()


    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
                # Cannot load, because in run.py, r<start_r is not allowed
                # all r in the loop, is not trained
                # I changed that in run.py to see effects
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            
            # data weighting
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            batch_timer = Timer()
            torch.cuda.synchronize()
            start = time.time()

            if hasattr(self, "begin_task"):
                self.begin_task(train_loader)

            if self.schedule_type == 'coswm': # step scheduler at each iter
                for epoch in range(self.config['schedule'][-1]):
                    self.epoch=epoch
                    self.model.prompt.cur_epoch_in_task = epoch
                    
                    # for param_group in self.optimizer.param_groups:
                    #     self.log('LR:', param_group['lr'])
                    num_batches = len(train_loader)
                    half_point = num_batches // 2
                    batch_timer.tic()
                    for i, (x, y, task)  in enumerate(train_loader):

                        # verify in train mode

                        # self.model.prompt.route_time_batch = 0.0
                        self.model.train()

                        # send data to gpu
                        if self.gpu:
                            x = x.cuda()
                            y = y.cuda()
                        
                        # model update
                        loss, output= self.update_model(x, y)
                        # ===== half-epoch flush =====
                        if self.model.prompt.rho_window_epochs == 0.5 and (i + 1) == half_point:
                            candidates = self._flush_rho_window(progress=0.5)

                            scope = self.model.prompt
                            if len(candidates) > 0:
                                next_l = min(candidates)

                                if (hasattr(scope, "epoch_cached_query_det") and next_l in scope.epoch_cached_query_det
                                        and len(scope.epoch_cached_query_det[next_l]) > 0):
                                    h = torch.cat(scope.epoch_cached_query_det[next_l], dim=0).to(
                                        next(self.model.parameters()).device)
                                    with torch.no_grad():
                                        mu = scope.compute_residual_init_key(next_l, h, q=0.95)
                                        scope.residual_init_key[next_l] = mu

                                scope.expand_now_layers.add(next_l)
                                scope.expand_layers.add(next_l)
                                scope.last_expand_epoch[next_l] = self.epoch
                                scope.no_improve_count[next_l] = 0

                                print(
                                    f"[Epoch {self.epoch}, Half] rho plateau -> expand another prompt at layer {next_l}")
                        self.scheduler.step()

                        # measure elapsed time
                        elapsed = batch_timer.toc()
                        batch_time.update(elapsed)

                        batch_timer.tic()

                        
                        # measure accuracy and record loss
                        y = y.detach()
                        accumulate_acc(output, y, task, acc, topk=(self.top_k,)) # already calculate train acc here? but logit range is narrow
                        losses.update(loss,  y.size(0)) 
                        batch_timer.tic()

                    self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                    self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                    # reset
                    losses = AverageMeter()
                    acc = AverageMeter()
                    if hasattr(self, "end_epoch"):
                        self.end_epoch()
                    torch.cuda.synchronize()
                    
            else:    
                for epoch in range(self.config['schedule'][-1]):
                    self.epoch=epoch
                    self.model.prompt.cur_epoch_in_task = epoch

                    if epoch > 0: self.scheduler.step()
                    # for param_group in self.optimizer.param_groups:
                    #     self.log('LR:', param_group['lr'])
                    batch_timer.tic()
                    for i, (x, y, task)  in enumerate(train_loader):

                        # verify in train mode
                        self.model.train()

                        # send data to gpu
                        if self.gpu:
                            x = x.cuda()
                            y = y.cuda()
                        
                        # model update
                        loss, output= self.update_model(x, y)

                        # measure elapsed time
                        batch_time.update(batch_timer.toc())  
                        batch_timer.tic()
                        
                        # measure accuracy and record loss
                        y = y.detach()
                        accumulate_acc(output, y, task, acc, topk=(self.top_k,)) # already calculate train acc here? but logit range is narrow
                        losses.update(loss,  y.size(0)) 
                        batch_timer.tic()

                    # eval update
                    self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                    self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                    # reset
                    losses = AverageMeter()
                    acc = AverageMeter()
                    if hasattr(self, "end_epoch"):
                        self.end_epoch()


        self.model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        if hasattr(self, "end_task"):
            self.end_task(train_loader)

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg, need_train
        except:
            return None, need_train

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):
        
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False):

        if model is None:
            model = self.model # why not pass the model in?

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()
        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                # Add torch.no_grad to save memory
                with torch.no_grad():
                    output = model.forward(input, target, use_general_prompt=getattr(self, "use_general_prompt", False))[:, :self.valid_out_dim]
                # output = model.forward(input)[:, :self.valid_out_dim]
                # TODO: try other task_metric?
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            else:
                # filter out class_ids of current task
                # task_in contains class_ids of current task, assume in ordered ascend
                # mask = target >= task_in[0] 
                mask = target >= min(task_in) 
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind] # select these samples from input data

                # mask = target < task_in[-1] # why not <=?
                # mask = target <= task_in[-1] # BUG
                mask = target <= max(task_in) # BUG
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if task_global:
                        # Add torch.no_grad to save memory
                        with torch.no_grad():
                            output = model.forward(input, target, use_general_prompt=getattr(self, "use_general_prompt", False))[:, :self.valid_out_dim]
                        # output = model.forward(input)[:, :self.valid_out_dim]
                        # TODO: try other task_metric?
                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                    else:
                        # Add torch.no_grad to save memory
                        with torch.no_grad():
                            output = model.forward(input, target, use_general_prompt=getattr(self, "use_general_prompt", False))[:, task_in]
                        # output = model.forward(input)[:, task_in]
                        # TODO: try other task_metric?
                        acc = accumulate_acc(output, target-min(task_in), task, acc, topk=(self.top_k,))
                        # acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
            
        model.train(orig_mode)

        if verbal:
            self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()

        save_dict = {
            "model": model_state
        }

        if hasattr(self.model, "prompt"):
            prompt_module = self.model.prompt


            save_dict["general_prompt_pool"] = getattr(prompt_module, "general_prompt_pool", {})
            save_dict["general_prompt_relations"] = getattr(prompt_module, "general_prompt_relations", {})
            save_dict["prompt_to_general"] = getattr(prompt_module, "prompt_to_general", {})
            save_dict["activated_prompts_history"] = getattr(prompt_module, "activated_prompts_history", {})


        self.log('=> Saving class model to:', filename)
        torch.save(save_dict, filename + 'class.pth')
        self.log('=> Save Done')

    # def expand_prompt_layer(self, prompt_module, layer_id, target_size):

    #     old_p = getattr(prompt_module, f"e_p_{layer_id}")
    #     old_k = getattr(prompt_module, f"e_k_{layer_id}")
    #
    #     n_old = old_p.shape[0]
    #     if target_size <= n_old:

    #
    #     device = old_p.device
    #     emb_d = old_p.shape[-1]
    #     p_len = old_p.shape[1]
    #     key_d = old_k.shape[-1]
    #
    #     new_p = torch.randn(target_size - n_old, p_len, emb_d, device=device) * 0.02
    #     new_k = torch.randn(target_size - n_old, key_d, device=device) * 0.02
    #     new_g = torch.zeros(target_size - n_old, device=device)
    #
    #     new_p = nn.Parameter(torch.cat([old_p, new_p], dim=0))
    #     new_k = nn.Parameter(torch.cat([old_k, new_k], dim=0))
    #     new_g = nn.Parameter(torch.cat([old_g, new_g], dim=0))
    #
    #     setattr(prompt_module, f"e_p_{layer_id}", new_p)
    #     setattr(prompt_module, f"e_k_{layer_id}", new_k)

    def expand_prompt_layer(self, prompt_module, layer_id, target_size):
        p_list = getattr(prompt_module, f"e_p_{layer_id}")  # ParameterList
        k_list = getattr(prompt_module, f"e_k_{layer_id}")

        n_old = len(p_list)
        if target_size <= n_old:
            return

        device = p_list[0].device
        emb_d = p_list[0].shape[-1]
        p_len = p_list[0].shape[0]
        key_d = k_list[0].shape[-1]

        n_new = target_size - n_old

        for _ in range(n_new):
            new_p = nn.Parameter(
                torch.randn(1, self.model.prompt.e_p_length, emb_d, device=device) * 0.02,
                requires_grad=False
            )
            new_k = nn.Parameter(
                torch.randn(1, key_d, device=device) * 0.02,
                requires_grad=False
            )

            p_list.append(new_p)
            k_list.append(new_k)

    def load_model(self, filename):
        # state_dict = torch.load(filename + 'class.pth')
        import torch, collections


        torch.serialization.add_safe_globals([collections.defaultdict])
        checkpoint = torch.load(filename + 'class.pth', map_location="cuda" if self.gpu else "cpu", weights_only=False)
        state_dict = checkpoint["model"]

        if hasattr(self.model, "prompt"):
            prompt_module = self.model.prompt


            layer_to_count = {}

            for name in state_dict.keys():
                if name.startswith("prompt.e_p_"):

                    parts = name.split(".")
                    layer_id = int(parts[1].split("_")[-1])
                    layer_to_count[layer_id] = max(
                        layer_to_count.get(layer_id, 0),
                        int(parts[2]) + 1
                    )


            for layer_id, target_size in layer_to_count.items():
                cur_size = len(getattr(prompt_module, f"e_p_{layer_id}"))
                if cur_size < target_size:
                    print(f"🧩 Expanding prompt layer {layer_id}: {cur_size} → {target_size}")
                    self.expand_prompt_layer(prompt_module, layer_id, target_size)

        # if hasattr(self.model, "prompt"):
        #     prompt_module = self.model.prompt
        #     for name, param in state_dict.items():
        #         if "prompt.e_p_" in name:
        #             layer_id = int(name.split("_")[-1])
        #             target_size = param.shape[0]
        #

        #             cur_p = getattr(prompt_module, f"e_p_{layer_id}")
        #             if cur_p.shape[0] != target_size:
        #                 print(f"🧩 Expanding prompt layer {layer_id}: {cur_p.shape[0]} → {target_size}")
        #                 self.expand_prompt_layer(prompt_module, layer_id, target_size)
        self.model.load_state_dict(state_dict, strict=False)

        if checkpoint is not None and hasattr(self.model, "prompt"):
            prompt_module = self.model.prompt
            prompt_module.general_prompt_pool = checkpoint.get("general_prompt_pool", {})
            prompt_module.general_prompt_relations = checkpoint.get("general_prompt_relations", {})
            prompt_module.prompt_to_general = checkpoint.get("prompt_to_general", {})
            prompt_module.activated_prompts_history = checkpoint.get("activated_prompts_history", {})
            print("Loaded general prompt structures.")
        # self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done from {}'.format(filename))
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()

    # sets model optimizers
    def init_optimizer(self):

        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim)

        return model

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]

    def predict(self, inputs):
        self.model.eval()
         # Add torch.no_grad to save memory
        with torch.no_grad():
            out = self.forward(inputs)
        # out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

class FinetunePlus(NormalNN):

    def __init__(self, learner_config):
        super(FinetunePlus, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter
