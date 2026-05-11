import torch
from model_enc_dec_visual import *
import numpy as np
import glob
import os
import sys
from collections import OrderedDict

'''
train_id = 1831011

ckpt_paths = [
f'/work/hdd/bdne/yzhang116/checkpoints_quijote/{train_id}/checkpoint_res_cross_entropy_epoch_0_step_43000_embed_384_batch_800_lrmax_0.0001_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'

]
'''

train_id = 1830812

ckpt_paths = [
f'/work/hdd/bdne/yzhang116/checkpoints_quijote/{train_id}/checkpoint_lm_cross_entropy_epoch_0_step_25000_embed_384_batch_800_lrmax_0.0001_lrmin_1e-05_layer_8_head_12_layervit_4_headvit_8_patch_4_dropout_0.0.pt'

]

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

avg_state_dict = OrderedDict()
num_ckpts = len(ckpt_paths)

for i, ckpt_path in enumerate(ckpt_paths):
    checkpoint = torch.load(ckpt_path, map_location=dev)
    state_dict = checkpoint["model"]
    print(checkpoint['loss'])

    # 去掉 DataParallel 的 module.
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if i == 0:
        # 初始化
        for k, v in state_dict.items():
            avg_state_dict[k] = v.clone()
    else:
        # 累加
        for k, v in state_dict.items():
            avg_state_dict[k] += v

for k in avg_state_dict:
    avg_state_dict[k] /= num_ckpts

HaloConfig = checkpoint['config']  # 用最后一个或第一个都行
pad_token = HaloConfig['pad_token']
# HaloConfig['flash'] = False  # Disable flash attention for visualization

model = HaloDecoderModel(HaloConfig).to(dev)
model.load_state_dict(avg_state_dict, strict=False)
model.eval()

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
dmo_dir = '/work/nvme/bdne/yzhang116/quijote_fields/train/'
halo_sentence_dir = '/work/nvme/bdne/yzhang116/quijote_halos/train/'

simid = 0
nsubbox = 5
dmo = np.load(dmo_dir+f'fields_{simid}.npy')
dmo = torch.from_numpy(dmo).to(dev).bfloat16()
dmo = torch.moveaxis(dmo, -1, 1)
dmo = dmo[:nsubbox]
sentences = np.load(halo_sentence_dir+f'halos_{simid}.npy')
sentences = torch.from_numpy(sentences).to(dev).long()
sentences = sentences[:nsubbox]

print('Data loading done!')
X = sentences[:, :-1]
Y = sentences[:, 1:].clone()
Y[:, :5] = pad_token
mask = torch.logical_not(X != pad_token)
masked_logits = torch.zeros(mask.shape, device=X.device, dtype=torch.float32)
MASK = masked_logits.masked_fill(mask, float('-inf'))[:,None,:]
with torch.no_grad():
    with ctx:
        all_self_atten, all_cross_atten, logits = model(X, dmo, 
                                    maskd=MASK, 
                                    targets=Y)
        print('Attention extraction done!')

# Save attention maps
all_self_atten = all_self_atten.cpu().numpy()
all_cross_atten = all_cross_atten.cpu().numpy()
logits = logits.cpu().numpy()
print(all_self_atten.shape)
print(all_cross_atten.shape)
print(logits.shape)
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/attention_self_{train_id}.npy', all_self_atten)
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/attention_cross_{train_id}.npy', all_cross_atten)
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/logits_{train_id}.npy', logits)

# save embedding weights
emb_weights = model.transformer.wte.weight.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/wte_weights_{train_id}.npy', emb_weights)
whe_weights = model.transformer.whe.weight.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/whe_weights_{train_id}.npy', whe_weights)
wprope_weights = model.transformer.wprope.weight.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/wprope_weights_{train_id}.npy', wprope_weights)
wce_weights = model.transformer.wce.weight.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/wce_weights_{train_id}.npy', wce_weights)

vit_emb = model.cnn3D.pos_embed.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/vit_pos_embed_{train_id}.npy', vit_emb)
vit_wce = model.cnn3D.wce.weight.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/vit_wce_weights_{train_id}.npy', vit_wce)
vit_wpe = model.cnn3D.wpe.weight.detach().cpu().numpy()
np.save(f'/work/hdd/bdne/yzhang116/att_matrix/vit_wpe_weights_{train_id}.npy', vit_wpe)