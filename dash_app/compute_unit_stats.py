import torch
import torch.nn as nn
from models.vgg_old import vgg11, vgg16, vgg16_bn
from tqdm import tqdm
from torchvision import transforms
import os
from data_utils import PatchDataset
import numpy as np
from netdissect import nethook
from netdissect import pbar, tally
from sklearn.manifold import TSNE

    
def resfile(resdir, f):
    return os.path.join(resdir, f)


def load_model(model='vgg16_bn', device=None, exp='vgg16_bn', grayscale=True, num_classes=2):
    if 'resnet' in model:
        model = eval(f'{model}(grayscale={grayscale})')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = eval(f'{model}(grayscale={grayscale})')
        model.classifier.fc8a = nn.Linear(model.classifier.fc8a.in_features, num_classes)
    
    model.load_state_dict(torch.load(f'../ckpt/{exp}/best_model.pth', map_location=torch.device(device)))
    
    model.eval()
    model = nethook.InstrumentedModel(model).eval()
    return model.to(device)


def load_dataset(exp='vgg16_bn', grayscale=True):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.Grayscale(num_output_channels=1 if grayscale else 3),
        transforms.ToTensor()
    ])
    test_data = f'../json/{exp}_result_128.json'
    target_dataset = PatchDataset(test_data, transform=transform, group=None)
    target_dataset.resolution = 128
    return target_dataset


def compute_topk(model, target_dataset, layername, resdir):
    pbar.descnext('topk')
    def compute_image_max(batch, *args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_batch = batch.to(device)
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.max(2)[0]

        return acts

    topk = tally.tally_topk(compute_image_max, target_dataset, sample_size=len(target_dataset),
            batch_size=32, num_workers=0, k=len(target_dataset), pin_memory=True, cachefile=resfile(resdir, 'topk.npz'))
    
    return topk


def compute_rq(model, target_dataset, layername, resdir, args):
    pbar.descnext('rq')
    # upfn = experiment.make_upfn(args, target_dataset, model, layername)
    def compute_samples(batch, *args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_batch = batch.to(device)
        _ = model(image_batch)
        acts = model.retained_layer(layername)

        return acts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

        # hacts = acts # upfn(acts)
        # return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    rq = tally.tally_quantile(compute_samples, target_dataset,
                              sample_size=len(target_dataset),
                              r=8192,
                              num_workers=0,
                              pin_memory=True,
                              cachefile=resfile(resdir, 'rq.npz'))
    return rq


def compute_act(model, dataloader, layername, device, resdir):
    cachefile = f'{resdir}/act.npy'
    if not os.path.exists(cachefile):
        acts = []
        for imgs, labels, preds, img_dirs in tqdm(dataloader):
            _ = model(imgs.to(device))
            batch_acts = model.retained_layer(layername)

            batch_acts = batch_acts.view(batch_acts.shape[0], batch_acts.shape[1], -1)

            batch_acts = batch_acts.mean(2) # batch_acts.max(2)[0]
            acts.append(batch_acts)

        acts = torch.cat(acts, dim=0).cpu()
        np.save(cachefile, acts.numpy())

    else:
        print(f'Loading cached {cachefile}')
        acts = np.load(cachefile)
        acts = torch.from_numpy(acts)

    return acts


def compute_act_quantile(quantile_table, acts, resdir):
    if acts.shape[0] != 1: 
        cachefile = f'{resdir}/act_quantile.npy'
        
        if not os.path.exists(cachefile):
            quantile_mat = torch.zeros(acts.shape[0], acts.shape[1])
            for i in range(acts.shape[0]): 
                for j in range(acts.shape[1]):
                    quantile_array = quantile_table[j, :]
                    upperbound = torch.where(quantile_array > acts[i][j])[0]
                    if upperbound.shape[0] != 0:
                        idx = torch.min(upperbound).item()
                    else:
                        idx = 1
                    quantile_mat[i][j] = idx / 100
            
            np.save(cachefile, quantile_mat.numpy())
        else:
            print(f'Loading cached {cachefile}')
            quantile_mat = np.load(cachefile)
    
    else:
        quantile_mat = torch.zeros(1, acts.shape[1])
        for j in range(acts.shape[1]):
            quantile_array = quantile_table[j,:]
            upperbound = torch.where(quantile_array > acts[0][j])[0]
            if upperbound.shape[0] != 0:
                idx = torch.min(upperbound).item()
            else:
                idx = 1
            quantile_mat[0][j] = idx / 100

        quantile_mat = quantile_mat.numpy()
    
    return quantile_mat


def iou_tensor(candidate: torch.Tensor, example: torch.Tensor):
    intersection = (candidate & example).float().sum((0, 1))
    union = (candidate | example).float().sum((0, 1))
    
    iou = intersection / (union + 1e-9)
    return iou.item()


def cluster_units(df, columns):
    tsne = TSNE(n_components=2, verbose=1, random_state=0, perplexity=30, n_iter=500)
    data_subset = df[columns].values
    tsne_results = tsne.fit_transform(data_subset)

    df['tsne-2d-one'] = (tsne_results[:,0] - tsne_results[:,0].min()) / (tsne_results[:,0].max() - tsne_results[:,0].min())
    df['tsne-2d-two'] = (tsne_results[:,1] - tsne_results[:,1].min()) / (tsne_results[:,1].max() - tsne_results[:,1].min())
    df['label'] = ['unknown' for _ in range(512)]
    df['act'] = [0 for _ in range(512)]
    df['unit'] = [i for i in range(512)]

    return df
