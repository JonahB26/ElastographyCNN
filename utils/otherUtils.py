import matplotlib.pyplot as plt
import numpy as np

def plot_results(pred, target, epoch, phase):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(pred[0].detach().cpu().squeeze(0).numpy(), cmap='viridis')
    plt.title(f'Prediction @ Epoch {epoch}')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(target[0].detach().cpu().squeeze(0).numpy(), cmap='viridis')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.savefig(f'{phase}_epoch_{epoch}.png')
    plt.close()


def compute_ncc(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    num = ((x - x_mean) * (y - y_mean)).sum()
    den = np.sqrt(((x - x_mean)**2).sum() * ((y - y_mean)**2).sum())
    return num / (den + 1e-8)