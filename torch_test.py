import numpy as np
import torch
import time
import torch.utils.benchmark as benchmark
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
    stmt='write(xyz_write, write_indices, write_buffer)',
    setup='from __main__ import write',
    globals={'xyz_write' : xyz_write, 'write_indices' : write_indices, 'write_buffer' : write_buffer})
print(t0.timeit((100)))
exit()

t0 = benchmark.Timer(
    stmt='load_multiple(xyz, load_indices)',
    setup='from __main__ import load_multiple',
    globals={'xyz' : xyz_load, 'load_indices' : load_indices})
print(t0.timeit((100)))