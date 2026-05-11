import torch

ckpt = torch.load("/work/hdd/bdne/yzhang116/checkpoints_quijote/1872236/checkpoint_cos_cross_entropy_epoch_0_step_7050_embed_384_batch_800_lrmax_1e-05_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt", map_location="cpu")

new_ckpt = {
    "model": ckpt["model"],
    "config": ckpt["config"]
}

torch.save(new_ckpt, "checkpoint_new.pt")