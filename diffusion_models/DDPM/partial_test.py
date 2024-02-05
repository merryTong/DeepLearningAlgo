import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import  torchvision.datasets as datasets

from ddpm import script_utils
from train_cifar import create_argparser

def test():
    args = create_argparser().parse_args()
    device = args.device
    args.batch_size = 1
    diffusion = script_utils.get_diffusion_from_args(args).to(device)


    train_dataset = datasets.CIFAR10(
            root='./mini_dataset',
            train=True,
            download=True,
            transform=script_utils.get_transform(),
        )
    train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
    )
    train_loader_iter = iter(train_loader)
    x, y = next(train_loader_iter)
    x = x.to(device)
    y = y.to(device)
    print("*"*100, y)

    for t in range(args.num_timesteps):
        noise = torch.randn_like(x)
        t = torch.tensor(t, dtype=torch.int64).reshape((args.batch_size, ))
        perturbed_x = diffusion.perturb_x(x, t, noise)        
        if t % 10 == 0:
            img = transforms.ToPILImage()(perturbed_x[0])
            img.save("noise_imgs/img_{}.jpg".format(t.item()))
        if t > 100: break

if __name__ == "__main__":
    test()