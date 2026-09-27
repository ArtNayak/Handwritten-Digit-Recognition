from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN, STD = 0.1307, 0.3081  # MNIST stats (reused for EMNIST, same 28x28 grayscale format)
NUM_CLASSES = {"mnist": 10, "emnist": 47}


def _orientation_fix(img):
    """EMNIST images are stored transposed relative to MNIST convention."""
    return transforms.functional.hflip(transforms.functional.rotate(img, -90))


def get_loaders(dataset="mnist", batch_size=128, data_dir="./data", augment=True):
    pre = [_orientation_fix] if dataset == "emnist" else []
    aug = [transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.1, 0.1))] if augment else []
    post = [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]

    train_tfm = transforms.Compose(pre + aug + post)
    test_tfm = transforms.Compose(pre + post)

    if dataset == "mnist":
        train = datasets.MNIST(data_dir, train=True, download=True, transform=train_tfm)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=test_tfm)
    elif dataset == "emnist":
        train = datasets.EMNIST(data_dir, split="balanced", train=True, download=True, transform=train_tfm)
        test = datasets.EMNIST(data_dir, split="balanced", train=False, download=True, transform=test_tfm)
    else:
        raise ValueError(dataset)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )
