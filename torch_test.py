import numpy as np
import torch
import time
import torch.utils.benchmark as benchmark


number_of_gaussians = 30_000_000
load_size = 3_000_000

dimensions = torch.tensor([3, 1, 3, 3, 4, 9], device='cuda')
all_gaussian_tensor =  torch.rand((number_of_gaussians, dimensions.sum()), device='cpu')
gaussian_tensors = [torch.rand((number_of_gaussians, dimensions[i]), device='cpu') for i in range(len(dimensions))]
load_indices = torch.randperm(number_of_gaussians)[:load_size]
def load_from_single(buffer, indices):
    gpu_buffer = buffer[indices].cuda()
    result1 = gpu_buffer[:, 0:3].cuda().contiguous()
    result2 = gpu_buffer[:, 3:4].cuda().contiguous()
    result3 = gpu_buffer[:, 4:7].cuda().contiguous()
    result4 = gpu_buffer[:, 7:10].cuda().contiguous()
    result5 = gpu_buffer[:, 10:14].cuda().contiguous()
    result6 = gpu_buffer[:, 10:23].cuda().contiguous()
    
    
    
def load_from_multiple(buffers, indices):
    gpu_buffers = []
    for b in buffers:
        gpu_buffers.append(b[indices].cuda()) 
    

storage_size = 10000000
load_size = 5000000
write_size = 5000000
load_indices = torch.randperm(storage_size)[:load_size]

xyz_load = torch.randn((storage_size, 3), device ='cpu').pin_memory()
xyz_write = torch.zeros((storage_size, 3), device ='cpu').pin_memory()
write_indices = torch.arange(0, write_size)# torch.randperm(storage_size, device ='cpu')[:load_size]
write_buffer = torch.randn((write_size, 3), device ='cuda')

def write(buffer, indices, write_data):
    xyz_write[indices] = write_data.cpu()

def load_multiple_non_blocking(buffer, indices):
    result1 = buffer[indices].cuda(non_blocking=True).contiguous()
    result2 = buffer[indices].cuda(non_blocking=True).contiguous()
    result3 = buffer[indices].cuda(non_blocking=True).contiguous()
    result4 = buffer[indices].cuda(non_blocking=True).contiguous()
    result5 = buffer[indices].cuda(non_blocking=True).contiguous()
    torch.cuda.synchronize()

def load_multiple(buffer, indices):
    result1 = buffer[indices].cuda().contiguous()
    result2 = buffer[indices].cuda().contiguous()
    result3 = buffer[indices].cuda().contiguous()
    result4 = buffer[indices].cuda().contiguous()
    result5 = buffer[indices].cuda().contiguous()
    
    
def load_single(buffer, indices):
    result1 = buffer[indices].cuda().contiguous()


t0 = benchmark.Timer(
    stmt='load_single(all_gaussian_tensor, load_indices)',
    setup='from __main__ import load_single',
    globals={'all_gaussian_tensor' : all_gaussian_tensor, 'load_indices' : load_indices})
print(t0.timeit((100)))

t0 = benchmark.Timer(
    stmt='load_from_multiple(gaussian_tensors, load_indices)',
    setup='from __main__ import load_from_multiple',
    globals={'gaussian_tensors' : gaussian_tensors, 'load_indices' : load_indices})
print(t0.timeit((100)))