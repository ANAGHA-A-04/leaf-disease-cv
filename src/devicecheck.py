import torch
print("PyTorch Version:", torch.__version__)
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
if cuda_available:
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("Device Name: CPU")