
import torch
from torch.utils.data.sampler import SubsetRandomSampler      
from torchvision import datasets, transforms  

# load training data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('...root/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)) # Normalize the dataset
                   ])), batch_size=1, shuffle=False)
# get indices of certain numbers (in here, 0, 1 and 2)
indices_zero = []
indices_one = []
indices_two = []
for idx, (data,target) in enumerate(train_loader):
    if target == 0:
        indices_zero.append(idx)
    if target == 1:
        indices_one.append(idx)
    if target == 2:
        indices_two.append(idx)
indices = indices_zero + indices_one + indices_two
# Convert list to torch.utils.data.sampler.SubsetRandomSampler object
sampler = SubsetRandomSampler(indices) 
mini_train_loader = torch.utils.data.DataLoader(datasets.MNIST('...root/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.2475,), (0.3776,))
                   ])),shuffle = False, sampler=sampler) # using sampler, shuffle must be false

for index, (data,target) in enumerate(mini_train_loader):
    # type your own code here
