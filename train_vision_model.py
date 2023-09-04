from PushTImageEnv import PushTImageEnv
from data_vision import PushTImageDataset, unnormalize_data, normalize_data
import torch as t
import torch
from torch import nn
import torchvision
import numpy as np
from conditional_unet1d import ConditionalUnet1D
from transformer_for_diffusion import TransformerForDiffusion
from diffusers.training_utils import EMAModel
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from tqdm.notebook import tqdm
import collections
import matplotlib.pyplot as plt

def train_model(
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        diffusion_timesteps=100,
        resnet="resnet18",
        device="cuda:0"
    )

    def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
        """
        name: resnet18, resnet34, resnet50
        weights: "IMAGENET1K_V1", None
        """
        # Use standard ResNet implementation from torchvision
        func = getattr(torchvision.models, name)
        resnet = func(weights=weights, **kwargs)

        # remove the final fully connected layer
        # for resnet18, the output dim should be 512
        resnet.fc = torch.nn.Identity()
        return resnet


    def replace_submodules(
            root_module: nn.Module, 
            predicate, 
            func) -> nn.Module:
        """
        Replace all submodules selected by the predicate with
        the output of func.

        predicate: Return true if the module is to be replaced.
        func: Return new module to use.
        """
        if predicate(root_module):
            return func(root_module)

        bn_list = [k.split('.') for k, m 
            in root_module.named_modules(remove_duplicate=True) 
            if predicate(m)]
        for *parent, k in bn_list:
            parent_module = root_module
            if len(parent) > 0:
                parent_module = root_module.get_submodule('.'.join(parent))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)
            tgt_module = func(src_module)
            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)
        # verify that all modules are replaced
        bn_list = [k.split('.') for k, m 
            in root_module.named_modules(remove_duplicate=True) 
            if predicate(m)]
        assert len(bn_list) == 0
        return root_module

    def replace_bn_with_gn(
        root_module: nn.Module, 
        features_per_group: int=16) -> nn.Module:
        """
        Relace all BatchNorm layers with GroupNorm.
        """
        replace_submodules(
            root_module=root_module,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//features_per_group, 
                num_channels=x.num_features)
        )
        return root_module

    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr"

    # parameters
    # pred_horizon = 16
    # obs_horizon = 2
    # action_horizon = 8
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )

    stats = dataset.stats

    # create dataloader
    dataloader = t.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True 
    )

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    noise_pred_net = noise_pred_net.to(device)

    nets = nn.ModuleList([vision_encoder, noise_pred_net]).to(device)

    ema = EMAModel(
        model=nets,
        power=0.75)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_timesteps,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    num_epochs = 100

    optimizer = t.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    losses = []
    for epoch in (e_iter := tqdm(range(num_epochs))):
        for batch in (b_iter := tqdm(dataloader, leave=False)):
            actions = batch["action"].to(device)
            images = batch["image"].to(device)
            pos = batch["agent_pos"].to(device)

            encoded_vectors = vision_encoder(images.reshape(-1, 3, 96, 96))
            encoded_vectors = encoded_vectors.reshape((images.shape[0], images.shape[1], -1))
            context = t.cat([encoded_vectors, pos], dim=-1).reshape(images.shape[0], -1)

            noise = t.randn(actions.shape, device=device)

            timesteps = t.randint(0, diffusion_timesteps, (len(actions),), device=device)
            noised_action = noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = noise_pred_net(noised_action, timesteps, global_cond=context)

            loss = t.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(noise_pred_net)
            b_iter.set_postfix({"Loss": loss.item()})
            losses.append(loss.item())
    return losses, ema