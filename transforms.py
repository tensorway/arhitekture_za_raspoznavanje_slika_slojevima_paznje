from torchvision import transforms
from torchvision import transforms as T 
from randomaug import RandAugment

vit_train_transform = transforms.Compose([
    RandAugment(2, 14),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

vit_val_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    
easy_transform = transforms.Compose([
    # transforms.Resize(32),
    transforms.RandomCrop(32),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomSolarize(threshold=200, p=0.3),
    # transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor(),
    transforms.RandomErasing(scale=(0.02, 0.14)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

reverse_normalize_transform = transforms.Compose([
    transforms.Normalize((0.0, 0.0, 0.0), (2, 2, 2)),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])