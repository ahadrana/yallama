import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
import seaborn as sns

BASE_PATH = "/tmp/tensor_debug"

def save_tensor(tensor, index, label):
    """
    Save a PyTorch tensor to a specified path using a label and index.
    
    Args:
    tensor (torch.Tensor): The tensor to save.
    index (int): The index to distinguish this tensor on disk.
    label (str): Label to categorize the tensor.
    path (str): Base directory to save the tensor.
    """
    # Construct the directory path with label
    directory_path = os.path.join(BASE_PATH, label)
    
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Construct the file path
    file_path = os.path.join(directory_path, f'tensor_{index}.pt')
    
    # Save the tensor
    tensor.requires_grad_(False)
    torch.save(tensor, file_path)

@torch.no_grad()
def main(label):
    # Construct the path to load tensors
    base_path = os.path.join(BASE_PATH, label)
    tensor1 = torch.load(os.path.join(base_path, 'tensor_0.pt')).cpu()
    tensor2 = torch.load(os.path.join(base_path, 'tensor_1.pt')).cpu()
    
    # Check if the tensors are close
    are_close = torch.allclose(tensor1, tensor2, atol=1e-5)
    print("All Close Returned:", are_close)
    
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()

    # Check if the tensors are close
    are_close_flat = torch.allclose(tensor1, tensor2, atol=1e-5)
    print("All Close Flat Returned:", are_close_flat)
    
    # compute the min dimension between the two tensors
    max_elements = max(tensor1.numel(), tensor2.numel())
    # calculate a row and column count for reshaping
    # round up 
    row_count = max_elements // 160
    
    if row_count * 160 < max_elements:
        padding_size = 160
        new_size = row_count * 160 + padding_size
        
        if tensor1.numel() < new_size:
            delta = new_size - tensor1.numel()
            tensor1 = torch.cat([tensor1, torch.zeros(delta)],-1)
        if tensor2.numel() < new_size:
            delta = new_size - tensor2.numel()
            tensor2 = torch.cat([tensor2, torch.zeros(delta)])
        row_count += 1
    
    # Check if the tensors are close
    are_close_append = torch.allclose(tensor1, tensor2, atol=1e-5)
    print("All Close Append Returned:", are_close_append)


    # Compute differences
    differences = (tensor1 - tensor2).abs()
    # Reshape differences into a rectangle
    differences = differences.reshape(row_count, 160)

    # Plot the heatmap of differences
    plt.figure(figsize=(12, 10))
    sns.heatmap(differences, cmap='bwr')
    plt.title('Heatmap of Differences for All 32,000 Elements')
    plt.savefig(os.path.join(base_path, 'heatmap.png'))
    



if __name__ == '__main__':

    # parse the arguments
    args = argparse.ArgumentParser()
    args.add_argument('--label', type=str, help='Tensors being debugged', required=True)
    args = args.parse_args()
    
    main(args.label)
