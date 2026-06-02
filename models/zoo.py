import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy
import torch.nn.functional as F
from collections import defaultdict
from torch import Tensor
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple

import time
import torch
# import matplotlib.pyplot as plt

class GateBackward(torch.autograd.Function):
    # jump the sign operation as the sign operation does not have gradients

    @staticmethod
    def forward(ctx: Any, scores: Tensor):
        signed_scores = torch.sign(scores)
        return signed_scores

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        return grad_output


def default_dict_set():
    return defaultdict(set)

class SmallAE(nn.Module):
    def __init__(self, dim, hid=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(),
                                    nn.Linear(hid, hid), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(),
                                    nn.Linear(hid, dim))

    def forward(self, x):
        return self.dec(self.enc(x))

# Our method
class SCOPE(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.key_d = key_dim
        self.soft_t = prompt_param[-1]
        self._init_smart(prompt_param)
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0, dtype=torch.float32)),
                                              requires_grad=False)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01, dtype=torch.float32)).item()
        self.general_prompt_pool = defaultdict(list)

        self.general_prompt_relations = defaultdict(dict)
        self.prompt_to_general = defaultdict(default_dict_set)
        self.activated_prompts_history = defaultdict(set)

        # ===== detector per layer per task =====
        self.detectors = nn.ModuleDict()  # key: "ae_l{l}_t{t}"
        self.detector_stats = {}  # (l,t) -> {"mu":..., "sigma":..., "thr":...}
        self.detector_input_dim = self.key_d  # 768 (x_querry dim)
        self.cached_query = {}  # dict: layer -> tensor[B, D]
        self.cached_query_route = {}
        self.expand_layers = set(self.e_layers)
        self.cur_task_id = -1
        self.cold_started = defaultdict(bool)  # layer -> bool
        self.new_prompt_indices = defaultdict(list)  # l -> [idx1, idx2, ...]
        self.cur_epoch_in_task = 0
        self.warmup_epochs = 10
        self.warmup_bias = 1.0

        # e prompt init
        for e in self.e_layers:
            setattr(self, f'e_p_{e}', nn.ParameterList([
                nn.Parameter(tensor_prompt(1, self.e_p_length, emb_d), requires_grad=True)
            ]))

            setattr(self, f'e_k_{e}', nn.ParameterList([
                nn.Parameter(tensor_prompt(1, self.key_d, ortho=True), requires_grad=True)
            ]))

    def _init_smart(self, prompt_param):

        # prompt basic param
        self.e_p_length = int(prompt_param[1])  # 8
        self.e_layers = [3,4,5,6,7]

    def update_prompt_params(self, opt, l, new_p, new_k):
        new_params = []
        if new_p.requires_grad:
            new_params.append(new_p)
        if new_k.requires_grad:
            new_params.append(new_k)

        if len(new_params) > 0:
            opt.param_groups[0]['params'].extend(new_params)

        print(f"[Optimizer Updated] Layer {l}: appended new prompt params")


    def _det_key(self, l, t):
        return f"ae_l{l}_t{t}"

    def init_detector(self, l, t):
        key = self._det_key(l, t)
        if key not in self.detectors:
            self.detectors[key] = SmallAE(self.detector_input_dim).to(next(self.parameters()).device)

    def freeze_detector(self, l, t):
        key = self._det_key(l, t)
        if key in self.detectors:
            for p in self.detectors[key].parameters():
                p.requires_grad = False
            self.detectors[key].eval()

    def detector_step(self, l, t, h, det_opt):
        self.init_detector(l, t)
        ae = self.detectors[self._det_key(l, t)]
        ae.train()
        x = h.detach()

        torch.cuda.synchronize()
        start = time.time()
        x_hat = ae(x)
        loss = F.mse_loss(x_hat, x)
        det_opt.zero_grad(set_to_none=True)
        loss.backward()
        det_opt.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        # print("TDE training:",elapsed)
        # print("AE loss:", loss.item())
        return loss.item()

    @torch.no_grad()
    def detector_score(self, l, t, h):
        key = self._det_key(l, t)
        ae = self.detectors[key]
        device = next(ae.parameters()).device
        h = h.to(device)
        ae.eval()
        x_hat = ae(h)
        err = ((x_hat - h) ** 2).mean(dim=1)  # [B]
        return err

    def finalize_detector_threshold(self, l, t, err_values, k_sigma=1.0):
        err = torch.as_tensor(err_values, device='cpu', dtype=torch.float32)
        mu = err.mean().item()
        sigma = err.std().item() + 1e-8
        thr = mu + k_sigma * sigma
        self.detector_stats[(l, t)] = {"mu": mu, "sigma": sigma, "thr": thr,
        "errs": err}

    def compute_residual_init_key(self, l, h, q=0.95):
        device = h.device
        K_list = getattr(self, f"e_k_{l}")

        if len(K_list) == 0:
            return F.normalize(h.mean(dim=0, keepdim=True), dim=1)

        K_old = torch.cat([k.to(device) for k in K_list], dim=0)
        K_old = F.normalize(K_old, dim=1)
        h_norm = F.normalize(h, dim=1)

        cos = torch.einsum("md,pd->mp", h_norm, K_old)
        max_cos = cos.max(dim=1).values
        residual = 1.0 - max_cos

        thr = torch.quantile(residual, q)
        idx = (residual >= thr).nonzero(as_tuple=False).squeeze(1)

        if idx.numel() == 0:
            mu = h_norm.mean(dim=0, keepdim=True)
        else:
            mu = h_norm[idx].mean(dim=0, keepdim=True)

        return F.normalize(mu, dim=1)

    def process_task_count(self):
        self.task_count += 1

    def add_prompt_params_to_optimizer(self, opt, params, lr=None):
        if lr is None:
            lr = opt.param_groups[0]["lr"]
        opt.add_param_group({
            "params": params,
            "lr": lr,
            "weight_decay": 0.0
        })

    def forward(self, x_querry, l, x_block, x_base, train=False, collect_tsne=False, use_general_prompt=False, task_id=None, opt=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:  # prompt location
            e_valid = True
            B, C = x_querry.shape
            x_querry1 = x_querry
            x_querry1 = F.normalize(x_querry1, dim=1)

            if train:
                patch_feats = x_block[:, 1:, :]  # [B, 196, 768]
                h0 = patch_feats.mean(dim=1)  # [B, 768]
                h0 = F.normalize(h0, dim=1)
                self.cached_query[l] = h0.detach()
                self.epoch_cached_query_det[l].append(h0.detach().cpu())
                self.cached_query_route[l] = x_querry1.detach()

            K_list = getattr(self, f'e_k_{l}')  # ParameterList
            p_list = getattr(self, f'e_p_{l}')

            torch.cuda.synchronize()
            route_start = time.time()

            K = torch.cat(list(K_list), dim=0)
            p = torch.cat(list(p_list), dim=0)
            n_K = nn.functional.normalize(K, dim=1)  # f, 768
            cos_sim = torch.einsum('bd,kd->bk', x_querry1,
                                   n_K)  # cosine similarity between batch images' cls token and prompts' keys
            logits = cos_sim / self.soft_t

            # ===== warmup bias only when expansion happened =====
            if train and getattr(self, "enable_warmup_bias", False):
                if self.cur_epoch_in_task < self.warmup_epochs:
                    new_idxs = self.new_prompt_indices.get(l, [])
                    if len(new_idxs) > 0:

                        t = self.cur_epoch_in_task / max(1, self.warmup_epochs - 1)
                        bias = (1.0 - t) * self.warmup_bias
                        logits[:, new_idxs] += bias

            alpha = torch.softmax(logits, dim=1)
            torch.cuda.synchronize()
            need_expand = False

            if train and (l in getattr(self, "expand_now_layers", set())):
                need_expand = True

                self.expand_now_layers.remove(l)

                if not self.cold_started[l]:
                    self.cold_started[l] = True


            elif train and (self.cur_task_id != 0) and (l in self.expand_layers) and (not self.cold_started[l]):
                need_expand = True
                self.cold_started[l] = True

            if need_expand:
                new_k = None
                print("there is one sample activate no prompts, maybe should add a new prompt")
                layer_feats = x_querry1
                inactive_samples = layer_feats

                if inactive_samples.size(0) > 0:
                    # ===== key init: use residual-aware init key from begin_task if available =====
                    if hasattr(self, "residual_init_key") and (l in self.residual_init_key):
                        RS = self.residual_init_key[l].to(x_querry1.device)  # [1, d]
                    else:
                        RS = inactive_samples.mean(dim=0, keepdim=True)
                        RS = F.normalize(RS, dim=1)

                    new_k = RS.clone()

                    base_idx = torch.argmax(alpha.mean(dim=0)).item()

                    base_p = p[base_idx:base_idx + 1].detach()  # [1, L, D]
                    new_p = base_p + 0.01 * torch.randn_like(base_p)  # very small noise
                    assert new_k is not None


                    new_k = nn.Parameter(new_k, requires_grad=True)
                    new_p = nn.Parameter(new_p, requires_grad=True)


                    getattr(self, f'e_k_{l}').append(new_k)
                    getattr(self, f'e_p_{l}').append(new_p)

                    new_idx = len(getattr(self, f'e_k_{l}')) - 1
                    self.new_prompt_indices[l].append(new_idx)

                    K_list = getattr(self, f'e_k_{l}')  # ParameterList
                    p_list = getattr(self, f'e_p_{l}')

                    K = torch.cat(list(K_list), dim=0)
                    p = torch.cat(list(p_list), dim=0)


                    if opt is not None:
                        self.add_prompt_params_to_optimizer(
                            opt,
                            [new_k, new_p],
                            lr=opt.param_groups[0]["lr"]
                        )

                    n_K = nn.functional.normalize(K, dim=1)  # f, 768
                    cos_sim = torch.einsum('bd,kd->bk', x_querry1,
                                           n_K)  # cosine similarity between batch images' cls token and prompts' keys
                    logits = cos_sim / self.soft_t

                    # ===== warmup bias only when expansion happened =====
                    if train and getattr(self, "enable_warmup_bias", False):
                        if self.cur_epoch_in_task < self.warmup_epochs:
                            new_idxs = self.new_prompt_indices.get(l, [])
                            if len(new_idxs) > 0:

                                t = self.cur_epoch_in_task / max(1, self.warmup_epochs - 1)
                                bias = (1.0 - t) * self.warmup_bias
                                logits[:, new_idxs] += bias

                    alpha = torch.softmax(logits, dim=1)
                p_a = torch.einsum('bk,kld->bld', alpha, p)

            else:
                p_a = torch.einsum('bk,kld->bld', alpha, p)

            # ===== compute residual for rho_l (trainable prompts only) =====
            if train and hasattr(self, "rho_sum"):

                h_route = self.cached_query_route[l]  # [B, d] (already normalized)
                K_list = getattr(self, f"e_k_{l}")
                trainable_idx = [i for i, k in enumerate(K_list) if k.requires_grad]

                if len(trainable_idx) > 0:
                    K_train = torch.cat([K_list[i] for i in trainable_idx], dim=0).to(h_route.device)  # [P_tr, d]
                    K_train = F.normalize(K_train, dim=1)
                    cos_tr = torch.einsum("bd,pd->bp", h_route, K_train)  # [B, P_tr]
                    max_cos = cos_tr.max(dim=1).values  # [B]
                    residual = 1.0 - max_cos  # [B]
                    self.rho_sum[l] += residual.sum().item()
                    self.rho_cnt[l] += residual.numel()

            P_ = p_a
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]
            Gk = Gv = None
            valid_general_indices = []
            loss = 0

        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
            if use_general_prompt:
                p_g_return = [Gk, Gv] if (Gk is not None and Gv is not None) else None
                return p_return, loss, x_block, p_g_return, valid_general_indices
            else:
                p_g_return = None
        else:
            p_return = None
            p_g_return = None
            valid_general_indices = []

        # return
        return p_return, loss, x_block, p_g_return, valid_general_indices


# @inproceedings{smith2023coda,
#   title={CODA-Prompt: COntinual decomposed attention-based prompting for rehearsal-free continual learning},
#   author={Smith, James Seale and Karlinsky, Leonid and Gutta, Vyshnavi and Cascante-Bonilla, Paola and Kim, Donghyun and Arbelle, Assaf and Panda, Rameswar and Feris, Rogerio and Kira, Zsolt},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={11909--11919},
#   year={2023}
# }
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more
        # fair in the spirit of continual learning and has little affect on performance
        #
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()  # clone trained prompt
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, y, train=False, task_id=None, opt=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = getattr(self, f'e_k_{l}')  # 100, 768
            A = getattr(self, f'e_a_{l}')  # 100, 768
            p = getattr(self, f'e_p_{l}')  # 100, 8, 768
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)  # start idx for 100 component
            f = int((self.task_count + 1) * pt)  # final idx for 100 component

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)  # f, 768
            q = nn.functional.normalize(a_querry, dim=2)  # bs, f, 768
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)  # bs, f (q k match)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)  # bs, 8, 768 reweighted p and sum along #component

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}', p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, emb_d, prompt_param):

        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0, 1]
        self.e_layers = [2, 3, 4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])  # self.n_tasks

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, y, train=False, task_id=None, opt=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self, f'e_k_{l}')  # 0 based indexing here
            p = getattr(self, f'e_p_{l}')  # 0 based indexing here

            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)

            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:, task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry), -1, -1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:, k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]

            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :i, :].reshape((B, -1, self.emb_d))
                Ev = P_[:, i:, :].reshape((B, -1, self.emb_d))
            else:
                i = int(self.e_p_length / 2)
                Ek = P_[:, :, :i, :].reshape(
                    (B, -1, self.emb_d))  # L2P, needs reshape top-k prompts into one longer prompt
                Ev = P_[:, :, i:, :].reshape(
                    (B, -1, self.emb_d))  # CODA-P avg several pre-defined task-specific components

        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length / 2)
            p = getattr(self, f'g_p_{l}')  # 0 based indexing here
            P_ = p.expand(len(x_querry), -1, -1)
            Gk = P_[:, :j, :]
            Gv = P_[:, j:, :]

        # combine prompts for prefix tuning
        if e_valid and g_valid:  # impossible for default setting; no overlap in layers
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0, 1, 2, 3, 4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, pretrained=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.pretrained = pretrained

        # get feature encoder
        if pt:
            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                          num_heads=12, ckpt_layer=0,
                                          drop_path_rate=0,  # num_classes=21843
                                          )
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']; del load_dict['head.bias']
            # zoo_model.load_state_dict(load_dict)

            if self.pretrained == "sup21k":
                dict_path = "pretrained/vit_base_patch16_224_augreg_in21k.bin"  # with head
                load_dict = torch.load(dict_path)
                del load_dict['head.weight'];
                del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
                print(f'Loading {self.pretrained} from {dict_path} ...')
            elif self.pretrained == "sup1k":
                dict_path = "pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin"  # with head
                load_dict = torch.load(dict_path)
                del load_dict['head.weight'];
                del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
                print(f'Loading {self.pretrained} from {dict_path} ...')
            elif self.pretrained == "ibot1k":
                dict_path = "pretrained/ibot-vit-base16.pth"  # ['state_dict']
                ckpt = torch.load(dict_path, map_location='cpu')['state_dict']  # with nead
                state_dict = zoo_model.state_dict()
                not_in_k = [k for k in ckpt.keys() if k not in state_dict.keys()]
                for k in not_in_k:
                    del ckpt[k]
                state_dict.update(ckpt)
                zoo_model.load_state_dict(state_dict)
                print(f'Loading {self.pretrained} from {dict_path} ...')
            elif self.pretrained == "dino1k":
                dict_path = "pretrained/dino_vitbase16_pretrain.pth"  # without head. blocks.0.att.qkv.weight
                load_dict = torch.load(dict_path, map_location='cpu')
                zoo_model.load_state_dict(load_dict)
                print(f'Loading {self.pretrained} from {dict_path} ...')
            else:
                print("Random Initialization")

        # classifier
        self.last = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1])
        elif self.prompt_flag == 'scope':
            self.prompt = SCOPE(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

    # pen: get penultimate features
    def forward(self, x, y, pen=False, train=False, return_pre_logits=False, cls_mean=None, optimizer=None,
                use_general_prompt=False):

        if self.prompt is not None:  # if having a prompt module
            with torch.no_grad():
                q, _, x_o, _ = self.feat(x, y=y)
                x_o = F.normalize(x_o, dim=1)
                x_o = x_o[:, 0, :]
                q = q[:, 0, :]  # extract each image's token
            if not use_general_prompt:
                out, prompt_loss, pre_logits, _ = self.feat(x, y=y, prompt=self.prompt, q=q, train=train,
                                                            task_id=self.task_id, opt=optimizer,
                                                            use_general_prompt=use_general_prompt)
                out_g = None
            else:
                out, prompt_loss, pre_logits, out_g = self.feat(x, y=y, prompt=self.prompt, q=q, train=train,
                                                                task_id=self.task_id, opt=optimizer,
                                                                use_general_prompt=use_general_prompt)
            out = out[:, 0, :]  # bs,197,768 -> bs,768 cls_token
            pre_logits = pre_logits[:, 0, :]
        else:
            out, _, pre_logits, _ = self.feat(x)
            out_g = None
            out = out[:, 0, :]
            pre_logits = pre_logits[:, 0, :]
        out = out.view(out.size(0), -1)
        pre_logits = pre_logits.view(pre_logits.size(0), -1)

        if return_pre_logits:
            return out

        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

    def analyze_prompt_diversity(self, e_layers):
        prompt_vecs = {}
        for l in e_layers:
            p = getattr(self, f"e_p_{l}")  # (num_prompts, prompt_length, emb_dim)
            p_mean = p.mean(dim=1)  # (num_prompts, emb_dim)
            prompt_vecs[l] = p_mean


        all_sims = []
        for li in e_layers:
            row = []
            for lj in e_layers:
                A = F.normalize(prompt_vecs[li], dim=1)
                B = F.normalize(prompt_vecs[lj], dim=1)
                sim = torch.matmul(A, B.T)
                row.append(sim.cpu().numpy())
            all_sims.append(np.concatenate(row, axis=1))
        sim_matrix = np.concatenate(all_sims, axis=0)


        print("=== 层间平均相似度 ===")
        for i, li in enumerate(e_layers):
            for j, lj in enumerate(e_layers):
                avg_sim = sim_matrix[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10].mean()
                print(f"Layer {li} vs Layer {lj}: {avg_sim:.4f}")


        plt.figure(figsize=(8, 8))
        plt.imshow(sim_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("跨层 Prompt Pool 相似度矩阵")
        plt.xlabel("Prompt Index Across Layers")
        plt.ylabel("Prompt Index Across Layers")
        plt.show()

        return sim_matrix

    def forward_fc(self, x):
        # x = self.feat.norm(x)
        out = self.last(x)
        return out

    @torch.no_grad()
    def _load_weights(self, model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
        """ Load weights from .npz checkpoints for official Google Brain Flax implementation
        """
        import numpy as np
        from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, \
            checkpoint_seq

        def _n2p(w, t=True):
            if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
                w = w.flatten()
            if t:
                if w.ndim == 4:
                    w = w.transpose([3, 2, 0, 1])
                elif w.ndim == 3:
                    w = w.transpose([2, 0, 1])
                elif w.ndim == 2:
                    w = w.transpose([1, 0])
            return torch.from_numpy(w)

        w = np.load(checkpoint_path)
        if not prefix and 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'

        if hasattr(model.patch_embed, 'backbone'):
            # hybrid
            backbone = model.patch_embed.backbone
            stem_only = not hasattr(backbone, 'stem')
            stem = backbone if stem_only else backbone.stem
            stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
            stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
            stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
            if not stem_only:
                for i, stage in enumerate(backbone.stages):
                    for j, block in enumerate(stage.blocks):
                        bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                        for r in range(3):
                            getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                            getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                            getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                        if block.downsample is not None:
                            block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                            block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                            block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
            embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        else:
            embed_conv_w = adapt_input_conv(
                model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
        model.patch_embed.proj.weight.copy_(embed_conv_w)
        model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
        if pos_embed_w.shape != model.pos_embed.shape:
            pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                model.pos_embed,
                getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        model.pos_embed.copy_(pos_embed_w)
        model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
        try:
            if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
                model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
                model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
        except:
            print('model does not contain head.')
        # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
        # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
        for i, block in enumerate(model.blocks.children()):
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

    def orth_loss(self, features, cls_mean):
        reg = 0.1
        if cls_mean:
            # orth loss of this batch
            sample_mean = []
            for k, v in cls_mean.items():
                if isinstance(v, list):
                    sample_mean.extend(v)
                else:
                    sample_mean.append(v)
            sample_mean = torch.stack(sample_mean, dim=0).to(features.device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(features.device))
            # print(loss)
            return reg * loss
        else:
            sim = torch.matmul(features, features.t()) / 0.8
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(0, sim.shape[0]).long().to(features.device))
            return reg * loss
            # return 0.


def vit_pt_imnet(out_dim, block_division=None, prompt_flag='None', prompt_param=None, pretrained=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param,
                  pretrained=pretrained)


if __name__ == "__main__":
    model = ViTZoo(pt=True)
