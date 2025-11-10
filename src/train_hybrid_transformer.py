import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from torch.cuda.amp import GradScaler, autocast
import wandb  # Add W&B import
from model_hybrid_transformer import HybridTransformerFlow

class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset for matrix format sequences with -100 padding and optional field conditioning.
    """

    def __init__(self, data_matrix: np.ndarray, fields: Optional[np.ndarray] = None, params: Optional[np.ndarray] = None):
        """
        Args:
            data_matrix: Matrix of shape [n_samples, max_blocks * 6] with -100 padding
            fields: Optional array of field embeddings [n_samples, n_field_tokens, d_model]
        """
        self.data = torch.FloatTensor(data_matrix).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.fields = None
        self.params = None
        if fields is not None:
            self.fields = torch.FloatTensor(fields).to('cuda' if torch.cuda.is_available() else 'cpu')
        if params is not None:
            self.params = torch.FloatTensor(params).to('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.fields is not None:
            if self.params is not None:
                return self.data[idx], self.fields[idx], self.params[idx]
            else:
                return self.data[idx], self.fields[idx]
        return self.data[idx], None


def collate_fn(batch: List) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Custom collate function for matrix format data."""
    if len(batch[0]) == 3:
        data = torch.stack([item[0] for item in batch])
        fields = [item[1] for item in batch if item[1] is not None]
        params = torch.stack([item[2] for item in batch if item[2] is not None])
        
        if fields:
            fields_tensor = torch.stack(fields)
            if params is not None:
                return data, fields_tensor, params
            return data, fields_tensor
        return data, None
    else:
        return torch.stack(batch), None


def train_model(
    model: HybridTransformerFlow,
    train_data_matrix: np.ndarray,
    train_fields: Optional[np.ndarray] = None,
    train_params: Optional[np.ndarray] = None,
    val_data_matrix: Optional[np.ndarray] = None,
    val_fields: Optional[np.ndarray] = None,
    val_params: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    best_model_name_prefix: str = "best_model",
    wandb_project: str = "half_gauss",  # Add W&B project name
    wandb_run_name: Optional[str] = None,  # Add W&B run name
    wandb_config: Optional[dict] = None  # Add extra config for W&B
):
    """
    Training loop for the hybrid model with matrix format data.
    """
    # Initialize W&B
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "device": device,
            "train_samples": len(train_data_matrix),
            "val_samples": len(val_data_matrix) if val_data_matrix is not None else 0,
            **(wandb_config or {})
        }
    )
    
    # Log model architecture
    wandb.watch(model, log="all", log_freq=100)
    
    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    dataset = SequenceDataset(train_data_matrix, train_fields, train_params)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # AdamW optimizer with modern hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95)  # Modern beta2 value
    )

    # Cosine annealing with warmup
    warmup_steps = min(500, len(dataloader) * 2)
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Mixed precision training
    scaler = GradScaler()
    
    # Training loop
    global_step = 0
    val_loss_min = 1e20
    for epoch in range(epochs):
        total_loss = 0
        total_flow_loss = 0
        total_decision_loss = 0
        n_batches = 0

        for batch_idx, (batch_data, batch_fields, batch_params) in enumerate(dataloader):
            # Data is already in matrix format, will be moved to device in forward()
            
            # Forward pass with mixed precision
            with autocast():
                if epoch > 2:
                    flow_weight = min(1.0, 0.1 + 0.1 * (epoch - 2))  # Gradually increase flow weight
                else:
                    flow_weight = 0.0
                # flow_weight = 0
                outputs = model(batch_data, vit_fields=batch_fields, params=batch_params, flow_weight=flow_weight)
                # outputs = model(batch_data, vit_fields=None, params=None)                
                loss = outputs['total_loss'] / gradient_accumulation_steps

            # outputs = model(batch_data, vit_fields=None, params=None)                
            # loss = outputs['total_loss'] / gradient_accumulation_steps


            # Backward pass
            scaler.scale(loss).backward()
            # loss.backward()

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients before clipping
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                # optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log to W&B every gradient accumulation step
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/global_step": global_step,
                    "train/learning_rate": current_lr,
                    "train/flow_weight": flow_weight,
                }, step=global_step)

            # Track losses
            total_loss += outputs['total_loss'].item()
            total_flow_loss += outputs['flow_log_prob'].item()
            total_decision_loss += outputs['decision_loss'].item()
            n_batches += 1
            
            # Log batch metrics to W&B
            wandb.log({
                "train/batch_total_loss": outputs['total_loss'].item(),
                "train/batch_flow_log_prob": outputs['flow_log_prob'].item(),
                "train/batch_decision_loss": outputs['decision_loss'].item(),
                "train/epoch": epoch,
            }, step=global_step)

            # if n_batches % 20 == 0:
            #     print(n_batches, total_loss, total_flow_loss, total_decision_loss)

        # Print progress
        avg_loss = total_loss / n_batches
        avg_flow = total_flow_loss / n_batches
        avg_decision = total_decision_loss / n_batches
        
        # Log epoch metrics to W&B
        wandb.log({
            "train/epoch_avg_loss": avg_loss,
            "train/epoch_avg_flow_log_prob": avg_flow,
            "train/epoch_avg_decision_loss": avg_decision,
            "train/epoch": epoch + 1,
        }, step=global_step)

        if avg_loss < val_loss_min:
            val_loss_min = avg_loss
            torch.save(model.state_dict(), f"/projects/bdne/spandey3/halo_gotham/GOTHAM/model_checkpoints/hybrid_flow/{best_model_name_prefix}_{wandb_project}.pth")
            print(f"  Saved best model with loss {val_loss_min:.4f}")

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Flow Log Prob: {avg_flow:.4f}")
            print(f"  Decision Loss: {avg_decision:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Validation if provided
            if val_data_matrix is not None:
                model.eval()
                with torch.no_grad():
                    val_batch_data = torch.FloatTensor(val_data_matrix[:batch_size])
                    val_batch_fields = None
                    if val_fields is not None:
                        val_batch_fields = torch.FloatTensor(val_fields[:batch_size])
                        val_batch_params = torch.FloatTensor(val_params[:batch_size])

                    val_outputs = model(val_batch_data, vit_fields=val_batch_fields, params=val_batch_params)
                    val_loss = val_outputs['total_loss'].item()
                    val_flow_loss = val_outputs['flow_log_prob'].item()
                    val_decision_loss = val_outputs['decision_loss'].item()
                    
                    print(f"  Validation Loss: {val_loss:.4f}")
                    
                    # Log validation metrics to W&B
                    wandb.log({
                        "val/total_loss": val_loss,
                        "val/flow_log_prob": val_flow_loss,
                        "val/decision_loss": val_decision_loss,
                        "val/epoch": epoch + 1,
                    }, step=global_step)
                    
                model.train()
    
    # Finish W&B run
    wandb.finish()
    
    return model

if __name__ == "__main__":
    D_MODEL = 72
    nparams = 6
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    import h5py as h5
    sdir = '/work/hdd/bdne/spandey3/quijote_data/halo_gotham_data/process_split'
    n_gpus = 16
    nrand_sel_box = 8192
    subsamp_ds = 4
    nvocab = 1
    DS_RES_POS_FAC = 1
    Mstar_cut = 12.7
    jdev = 0
    savefname = f'{sdir}/FLOW_SPLIT_HALO_DATA_{n_gpus}_gpus_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}_grid_{nvocab//DS_RES_POS_FAC}_xyzMvc_{Mstar_cut}.h5'
    df = h5.File(savefname, 'r')
    train_data_matrix = torch.tensor(df[f'story_train_dev_{jdev}'][:]).to(torch.float32).to(device)
    train_params_matrix = torch.tensor(df[f'params_train_dev_{jdev}'][:]).to(torch.float32).to(device)

    val_data_matrix = torch.tensor(df[f'story_val_dev_{jdev}'][:]).to(torch.float32).to(device)
    val_params_matrix = torch.tensor(df[f'params_val_dev_{jdev}'][:]).to(torch.float32).to(device)

    df.close()

    sdir = '/work/hdd/bdne/spandey3/quijote_data/halo_gotham_data/process_split'
    grid_sbox = 8
    savefname = f'{sdir}/SPLIT_DMO_DATA_{n_gpus}_gpus_density3Dgrid_{grid_sbox}_isim_all_nrandsubsel_{int(nrand_sel_box/subsamp_ds)}.h5'

    with h5.File(savefname, 'r') as f:
        ind_all_train = np.arange(f[f'dm_train_dev_{jdev}'][:].shape[0])
        ind_all_val = np.arange(f[f'dm_val_dev_{jdev}'][:].shape[0])
        ind_sel_train = ind_all_train
        ind_sel_val = ind_all_val

        print(f"ind_sel_train = {ind_sel_train.shape}, ind_sel_val = {ind_sel_val.shape}", flush=True)
        dm_train_gpu = torch.tensor(f[f'dm_train_dev_{jdev}'][:][ind_sel_train]).to(torch.float32).to(device)
        dm_val_gpu = torch.tensor(f[f'dm_val_dev_{jdev}'][:][ind_sel_val]).to(torch.float32).to(device)
        grid_size = int(f['grid'][()])
    f.close()



    # Initialize model with modern architecture
    tail_bound_min = 0.0
    tail_bound_max = 1.0
    input_dim = 8
    print("Initializing model with modern architecture (RMSNorm, bias-free, GroupNorm)...")

    config_dict = {
        "input_dim": input_dim,
        "d_model": D_MODEL,
        "n_heads": 8,
        "n_layers": 4,
        "d_ff": 256,
        "flow_layers": 2,
        "flow_hidden": 128,
        "n_bins": 16,
        "tail_bound_min": tail_bound_min,
        "tail_bound_max": tail_bound_max,
        "dropout_val": 0.1,
        "n_groups": 8,
        "pad_value": -1.0,
        "max_blocks": train_data_matrix.shape[1]//input_dim,
        'nparams': train_params_matrix.shape[1], 
        'ninp_density': dm_train_gpu.shape[1], 
        'patch_size':1,
        'n_layers_vit': 1, 'n_heads_vit': 4,
        'layers_types':['res_cbam'],
    }


    model = HybridTransformerFlow(
        config_dict
    )

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    indsel_train = 131072
    indsel_val = 2048*8
    print("\nStarting training with modern optimization...")
    
    # Prepare W&B config
    wandb_config = {
        **config_dict,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "indsel_train": indsel_train,
        "indsel_val": indsel_val,
        "Mstar_cut": Mstar_cut,
        "grid_sbox": grid_sbox,
        "nrand_sel_box": nrand_sel_box,
        "subsamp_ds": subsamp_ds,
    }
    
    trained_model = train_model(
        model,
        train_data_matrix[:indsel_train],
        train_fields=dm_train_gpu[:indsel_train],
        train_params=train_params_matrix[:indsel_train],
        val_data_matrix=val_data_matrix[:indsel_val],
        val_fields=dm_val_gpu[:indsel_val],
        val_params=val_params_matrix[:indsel_val],
        epochs=80,
        batch_size=2048*2,    
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        wandb_project="hybrid-transformer-flow",  # Change this to your project name
        wandb_run_name=f"run2_Mstar{Mstar_cut}_lr1e-4_bs{2048*2}",  # Optional: customize run name
        wandb_config=wandb_config
    )