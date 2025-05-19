# -*- coding: utf-8 -*-


!git clone 

!pip install -U datasets



import datasets
import torch

dataset = datasets.load_dataset('synthseq/flipflop')

def tokenize_batch(batch):
    mapping = {'w': 0, 'r': 1, 'i': 2, '0': 3, '1': 4}
    tokenized_batch = [[mapping[char] for char in s] for s in batch['text']]
    return {'text': tokenized_batch}

dataset.set_transform(tokenize_batch)

val_loader = torch.utils.data.DataLoader(dataset['val']['text'], batch_size=32, shuffle=True)
val_dense_loader = torch.utils.data.DataLoader(dataset['val_dense']['text'], batch_size=32, shuffle=True)
val_sparse_loader = torch.utils.data.DataLoader(dataset['val_sparse']['text'], batch_size=32, shuffle=True)

!cd unlocking_state_tracking/; chmod +x setup.sh; ./setup.sh

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf

config_yaml= '/content/unlocking_state_tracking/xlstm/experiments/parity_mamba_single_layer.yaml'
cfg = OmegaConf.load(config_yaml)

import sys

sys.path.append('/content/unlocking_state_tracking/xlstm/mamba')

sys.path.append('/content/unlocking_state_tracking/flash-linear-attention_mod')

from fla.models.mamba_py.mamba import Mamba, MambaConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

mamba_1_layer_16_dim = nn.Sequential(Mamba(from_dict(MambaConfig, OmegaConf.to_container(cfg.model))).to(device=cfg.training.device), nn.Linear(16, 2))

device = 'cuda'

def embed(labels, dim):
    # labels.shape = (batch, seq. len)
    return F.embedding(labels, torch.eye(dim).to(device))

def model_eval(model, labels, embed_dim, to_cuda=False):
    # batch.shape = (batch, seq. len)

    if to_cuda:
        labels = labels.to(device)

    embed_inputs = embed(labels, embed_dim)

    outputs = model(embed_inputs)

    class_labels = torch.max(labels[:, 1:] - 3, torch.tensor([0]).to(device))
    class_logits = outputs[:, :-1, :].permute(0, 2, 1)
    class_pred = torch.argmax(class_logits, dim=1)

    mask = (labels == 1)[:, :-1]


    loss = torch.nn.CrossEntropyLoss(reduction='none')(class_logits, class_labels)
    loss_masked = loss * mask
    if mask.sum() < 1:
        err = 0
    else:
        err = (class_pred[mask] != class_labels[mask]).sum() / mask.sum()
    return loss_masked, err

def model_train(model, optimizer, data_gen, valid_names= [], valid_loaders=[], embed_dim=5, valid_size=1000, valid_batch=32, to_cuda = False, valid_report_period = 100):
    running_err = 0.
    valids_errs = {valid_name:[] for valid_name in valid_names}
    for i, data in enumerate(data_gen):
        data = data.permute(1, 0)
        loss_masked, err = model_eval(model, data, embed_dim=embed_dim, to_cuda=to_cuda)
        running_err += err

        optimizer.zero_grad()
        loss_masked.backward(torch.ones((loss_masked.shape)).to(device))
        optimizer.step()

        if i % valid_report_period == valid_report_period-1:
            valid_datas = [[batch for _, batch in zip(range(valid_size//valid_batch), valid_loader)] for valid_loader in valid_loaders]
            for idx, valid_data in enumerate(valid_datas):
                valid_run_err = 0.
                for vdata in valid_data:
                    vdata = torch.stack(vdata).permute(1,0)
                    _, verr = model_eval(model, vdata, embed_dim=embed_dim, to_cuda=True)
                    valid_run_err += verr
                valids_errs[valid_names[idx]].append(valid_run_err / len(valid_data))
            print_errs= "   ".join([f"{valid_name}: {valid_errs[-1]}" for valid_name, valid_errs in valids_errs.items()])
            if all(valid_errs[-1] == torch.tensor(0.) for valid_errs in valids_errs.values()):
                return valids_errs
            print(f'   batch {i+1}    running err: {running_err/(valid_report_period)}   {print_errs}')
            running_err = 0.
    return valids_errs

training_loader =  torch.utils.data.DataLoader(dataset['train']['text'], batch_size=32, shuffle=True)

training_loader =  torch.utils.data.DataLoader(dataset['train']['text'], batch_size=32, shuffle=True)
train_data_tensor = torch.stack([torch.stack(batch)[:64, :] for batch, _ in zip(training_loader, range(16000))]).to('cuda')

model = nn.Sequential(Mamba(from_dict(MambaConfig, OmegaConf.to_container(cfg.model))), nn.Linear(16, 2)).to(device)

total_val_sparse_errs = []
total_val_errs = []
total_val_dense_errs = []

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
for epoch in range(2):
    valids_errs = model_train(model, optimizer, train_data_tensor,
                            valid_names= [ 'val', 'val_sparse', 'val_dense'],
                            valid_loaders=[val_loader, val_sparse_loader, val_dense_loader], embed_dim=16,
                            valid_size=2000, valid_batch=32, to_cuda = False, valid_report_period = 100)
    total_val_errs += valids_errs['val']
    total_val_sparse_errs += valids_errs['val_sparse']
    total_val_dense_errs += valids_errs['val_dense']

    if all([valid_errs[-1] == torch.tensor(0.) for valid_errs in valids_errs.values()]):
        break


valid_run_err = 0.
for vdata in val_sparse_loader:
    vdata = torch.stack(vdata).permute(1,0)
    _, verr = model_eval(model, vdata, embed_dim=16, to_cuda=True)
    valid_run_err += verr
test_sparse_errs = (valid_run_err/len(val_sparse_loader))

val_sparse_errs = torch.stack(total_val_sparse_errs).cpu().detach().numpy()
val_errs = torch.stack(total_val_errs).cpu().detach().numpy()
val_dense_errs = torch.stack(total_val_dense_errs).cpu().detach().numpy()

test_sparse_errs

import torch.nn.functional as F

def embed_input( input):
    labels = input.permute(1, 0)
    return embed(labels, 16)

def example_states(model, d_state, batch, length, x):

    x_old = x
    x = embed_input(x)

    dim, d_conv, expand = 16, 4, 2
    ssm_states = []
    outs = []
    # for i in range(length):
    #     out = model(x[:, i : i + 1, :])
    #     infer_params.seqlen_offset += 1
    #     outs.append(out)
    cache = (None, torch.zeros(batch, expand*dim, d_conv-1).to(device))
    for i in range(length):
        out, cache = model.step(x[:, i : i + 1, :].view(batch, dim), cache)
        ssm_state, _ = cache
        outs.append(out.cpu())
        ssm_states.append(ssm_state.cpu())
    states = torch.stack(ssm_states).cpu()
    outs = torch.stack(outs).cpu()

    return states, x_old, outs

def get_state(labels):
    states = [torch.tensor(0)]
    actions = [(labels[2*i], labels[2*i+1]) for i in range(len(labels)//2)]
    for act, val in actions:
        if act == 0:
            states += [val-3, val-3]
        else:
            last = states[-1]
            states += [last, last]
    return states[:-1]

model[0].layers[0].mixer

import matplotlib.pyplot as plt

mapping = {'w': 0, 'r': 1, 'i': 2, '0': 3, '1': 4}
x = 'w0w1r1i1r1r1i1w0i0r0r0w1w1r1w1r1w1r1w1w1r1r1r1w0w0w1r1r1r1r1r1r1w0r0r0i0r0w1w0w0w1r1w0r0w1w0w1w0r0i1w1r1r1w0r0w1r1w0r0w0w1r1r1r1r1r1r1w0r0w0w1r1w1w1w1w1w1r1r1w1r1r1i0w1r1r1r1r1w0r0w0w0r0w0w1w0i0i1r0r0r0i1r0w1i0w1w1w1i1w1r1i1i1w0i0w0r0r0w1w0w0r0i1w1w0w0w1w1r1w0w1r1w1r1r1w0r0r0r0r0w1r1i1r1w0w0i1r0w0r0w1r1w1r1r1r1w0w1w0w1w0r0r0w0w0r0w1w0r0w0r0r0w0w0w1r1r1w1r1w1r1r1r1w1r1r1r1i0w1w0r0i0r0w1w1i1i0w1r1r1r1r1w1i1r1r1w1w1r1i1i1r1w1r1r1r1r1w1w0r0i0r0w0w0r0i0w1i1r1r1w1w0r0r0w0r0r0r0w1r1w1r1w0r0r0w1w1w0r0w1i0i0r1r1r1r1'
x = torch.tensor([mapping[c] for c in x]).unsqueeze(-1).to('cuda')

states, labels,_ = example_states(model[0].layers[0].mixer, 16, 1, 512, x)
print(states.shape)
b = 0



states_proj = []
for c in range(10):
    _, _, states_pca = torch.pca_lowrank(states[:,b,c,:])
    states_proj.append(torch.matmul(states[:,b,c,:], states_pca[:, :2]))

labels = [get_state(list(labels.cpu())) for labels in labels.permute(1, 0)]


cols = ['g' if i == 0 else 'r' if l == 0 else 'b' for i, l in enumerate(labels[b])]

fig, ax = plt.subplots(2, 5)
fig.set_figwidth(15)
for c in range(2):
    for r in range(5):
        ax[c][r].scatter(states_proj[5*c+r][:,0].detach().numpy(), states_proj[5*c+r][:,1].detach().numpy(), c=cols, s=0.5)

fig.suptitle("PCA of states for each channel; no schedule; val_dense input (red=0, blue=1)")

x1 = 'w0' + ''.join(["i0" for i in range(510)]) + 'r0'
x1 = torch.tensor([mapping[c] for c in x1]).unsqueeze(-1).to('cuda')
states1, labels1, outs1 = example_states(model[0].layers[0].mixer, 16, 1, 1024, x1)
labels1 = [get_state(list(labels.cpu())) for labels in labels1.permute(1, 0)]

x2 = 'w1' + ''.join(["i0" for i in range(510)]) + 'r1'
x2 = torch.tensor([mapping[c] for c in x2]).unsqueeze(-1).to('cuda')
states2, labels2, outs2 = example_states(model[0].layers[0].mixer, 16, 1, 1024, x2)
labels2 = [get_state(list(labels.cpu())) for labels in labels2.permute(1, 0)]

states = torch.cat([states1, states2], dim=0)
labels = labels1[0] + labels2[0]
print(len(labels))
states_proj = []
for c in range(10):
    _, _, states_pca = torch.pca_lowrank(states[:,b,c,:])
    states_proj.append(torch.matmul(states[:,b,c,:], states_pca[:, :2]))



cols = ['g' if i == 0 else 'r' if l == 0 else 'b' for i, l in enumerate(labels)]

fig, ax = plt.subplots(2, 5)
fig.set_figwidth(15)

for c in range(2):
    for r in range(5):
        ax[c][r].scatter(100*states_proj[5*c+r][:,0].detach().numpy(), 100*states_proj[5*c+r][:,1].detach().numpy(), c=cols, s=0.5)
        ax[c][r].xaxis.set_tick_params(labelsize=6)
        ax[c][r].yaxis.set_tick_params(labelsize=6)
#fig.suptitle("PCA of states for each channel (red=high state input, blue=low state input), input length 2024", fontsize=16)



embed_x1 = embed(x1, 16).permute(1,0,2)
out1 = model(embed_x1)

class_logits = out1[:, :-1, :].permute(0, 2, 1)
class_pred1 = torch.argmax(class_logits, dim=1)

print(class_pred1[0,:-1])

print(class_pred2[0,:-1])

embed_x2 = embed(x2, 16).permute(1,0,2)
out2 = model(embed_x2)

class_logits = out2[:, :-1, :].permute(0, 2, 1)
class_pred2 = torch.argmax(class_logits, dim=1)

plt.plot((class_pred1[0,:] - class_pred2[0,:]).detach().cpu())

print(class_pred)

x3 = 'w1w0w1w0w0i1r0r0'
x3 = torch.tensor([mapping[c] for c in x3]).unsqueeze(0).to('cuda')

embed_x3 = embed(x3, 16)
out3 = model(embed_x3)
print(out3.shape)
class_logits = out3[:, :-1, :].permute(0, 2, 1)
class_pred = torch.argmax(class_logits, dim=1)

print(class_pred)

test_input = torch.stack([torch.tensor([0,3,2,4,2,4,2,4,1,0,4,1,4,1])]).to('cuda')
test_input_embed = embed(test_input, 16)
output = model(test_input_embed)
print(output.shape)
class_logits = output[:, :, :].permute(0, 2, 1)
class_pred = torch.argmax(class_logits, dim=1)
print(class_pred)

