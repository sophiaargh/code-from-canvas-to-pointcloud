import numpy as np
import matplotlib.pyplot as plt
import re

def read_pfm(filepath):
    with open(filepath, 'rb') as f:
        # read header line by line
        header = f.readline().decode('latin-1').strip()
        assert header in ('PF', 'Pf'), f"Not a PFM file: {header}"
        
        dims = f.readline().decode('latin-1').strip()
        W, H = map(int, dims.split())
        
        scale = float(f.readline().decode('latin-1').strip())
        endian = '<' if scale < 0 else '>'
        
        # read the rest as raw bytes — no skipping
        data = np.frombuffer(f.read(), dtype=np.dtype(endian + 'f'))
    
    data = data.reshape((H, W))[::-1].copy()  # flip vertical
    return data

def show_depth_error(pred_depth, gt_path, view_idx=0):
    """
    pred_depth: (3, H, W) numpy array from VGGT
    gt_path: path to .pfm file for this view
    """
    gt = read_pfm(gt_path)                    # (H, W)
    pred = pred_depth[view_idx]               # (H, W)

    # resize pred to match gt if needed
    if pred.shape != gt.shape:
        from skimage.transform import resize
        pred = resize(pred, gt.shape, anti_aliasing=True)

    # valid mask — ignore zero/invalid gt pixels
    mask = gt > 0

    # scale alignment (VGGT is affine-invariant)
    scale = np.median(gt[mask]) / np.median(pred[mask])
    pred_aligned = pred * scale

    # metrics
    abs_rel = np.mean(np.abs(pred_aligned[mask] - gt[mask]) / gt[mask])
    rmse    = np.sqrt(np.mean((pred_aligned[mask] - gt[mask]) ** 2))
    print(f"AbsRel: {abs_rel:.4f}  |  RMSE: {rmse:.4f}")

    # error map
    error = np.abs(pred_aligned - gt)
    error[~mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#0e0e0e')
    titles = ['GT depth', 'Predicted depth (aligned)', 'Absolute error']
    maps   = [gt, pred_aligned, error]
    cmaps  = ['magma', 'magma', 'hot']

    for ax, data, title, cmap in zip(axes, maps, titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, color='#aaa', fontsize=10)
        ax.axis('off')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='#aaa')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaa')

    plt.suptitle(f'AbsRel: {abs_rel:.4f}  |  RMSE: {rmse:.4f}',
                 color='white', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'depth_error_view{view_idx}.png', dpi=150,
                bbox_inches='tight', facecolor='#0e0e0e')
    plt.show()
    
    return {"AbsRel": abs_rel, "RMSE": rmse}

#Example usage:
depth = predictions["depth"].squeeze(0)[..., 0].cpu().numpy()  # (3, H, W)
metrics = show_depth_error(depth, gt_path="BlendedMVS/5a3ca9cb270f0e3f14d0eddb/rendered_depth_maps/00000000.pfm", view_idx=0)

def read_pfm(filepath):
    with open(filepath, 'rb') as f:
        # read header line by line
        header = f.readline().decode('latin-1').strip()
        assert header in ('PF', 'Pf'), f"Not a PFM file: {header}"
        
        dims = f.readline().decode('latin-1').strip()
        W, H = map(int, dims.split())
        
        scale = float(f.readline().decode('latin-1').strip())
        endian = '<' if scale < 0 else '>'
        
        # read the rest as raw bytes — no skipping
        data = np.frombuffer(f.read(), dtype=np.dtype(endian + 'f'))
    
    data = data.reshape((H, W))[::-1].copy()  # flip vertical
    return data