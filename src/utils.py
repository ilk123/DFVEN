import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.colors as colors
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def retrieve_files(dir, suffix='png|jpg'):
    """ retrive files with specific suffix under dir and sub-dirs recursively
    """

    def retrieve_files_recursively(dir, file_lst):
        for d in sorted(os.listdir(dir)):
            dd = osp.join(dir, d)

            if osp.isdir(dd):
                retrieve_files_recursively(dd, file_lst)
            else:
                if osp.splitext(d)[-1].lower() in ['.' + s for s in suffix]:
                    file_lst.append(dd)

    if not dir:
        return []

    if isinstance(suffix, str):
        suffix = suffix.split('|')

    file_lst = []
    retrieve_files_recursively(dir, file_lst)
    file_lst.sort()

    return file_lst

def count_param(model):
    for name, module in model.named_modules():
        if not list(module.parameters()):
            continue
        num_params = sum(p.numel() for p in module.parameters())
        print(f'{name}: {num_params} parameters')

def save_image(img_path, img_data, to_bgr=False):
    if to_bgr:
        img_data = img_data[..., ::-1]  # rgb2bgr

    os.makedirs(osp.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, img_data)

def save_seqence(seq_dir, seq_data, frm_idx_lst=None, to_bgr=False):
    if to_bgr:
        seq_data = seq_data[..., ::-1]  # rgb2bgr
    
    tot_frm = len(seq_data)
    if frm_idx_lst is None:
        frm_idx_lst = ['{:04d}.png'.format(i) for i in range(tot_frm)]
    
    os.makedirs(seq_dir, exist_ok=True)
    for i in range(tot_frm):
        cv2.imwrite(osp.join(seq_dir, frm_idx_lst[i]), seq_data[i])

def show_tensor(tensor):
    if len(tensor.size()) == 4:
        _tensor = tensor[0, ...]
    numpy = _tensor.permute(1, 2, 0).cpu().numpy()
    plt.imshow(numpy)
    plt.show(block=True)

def show_degrade(embedding):
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    x_tsne = tsne.fit_transform(np.array(embedding['embedding']))
    c = np.array(embedding['param'])[:, 0]
    plt.figure(figsize=(10, 8))
    color_list = ['red', 'green', 'blue', 'yellow']
    cmap = colors.ListedColormap(color_list)
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1], marker='o', c=c, cmap=cmap, edgecolors='none', s=30, label=c)
    plt.legend(*scatter.legend_elements())
    plt.grid(True)
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.array(embedding['embedding']))
    plt.figure(figsize=(10, 8))
    color_list = ['red', 'green', 'blue', 'yellow']
    cmap = colors.ListedColormap(color_list)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=c, cmap=cmap, s=20, label=c)
    plt.legend(*scatter.legend_elements())
    plt.grid(True)
    plt.show()

def save_loss(file_path, iter_num, Loss):
    with open(file_path, 'a+') as f:
        f.write(f'{iter_num}: \n')
        for k, v in Loss.items():
            f.write(f'{k}: {v}\n')
