import argparse
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

from .models.ConvHDC import ConvHDC
from .models.ResConvHDC import ResConvHDC

parser = argparse.ArgumentParser()
parser.add_argument('--model-architecture', '-ma', default='ConvHDC',
                    choices=['ConvHDC', 'ResConvHDC'], help='model architecture: ConvHDC or ResConvHDC (default: resnet32)')
parser.add_argument('--dataset', '-ds', default='MNIST',
                    choices=['MNIST', 'FashionMNIST', 'CIFAR10'], help='dataset for training (default: MNIST)')
parser.add_argument('--dataset-download-path', '-ddp', default='../data',
                    help='path to download the data to (default: ../data)')
parser.add_argument('--epochs', '-ep', default=200, type=int,
                    help='number of epochs to train for (default: 200)')
parser.add_argument('--dimension', '-D', default=10000, type=int,
                    help='hypervector dimension (default: 10000)')
parser.add_argument('--weighted-bundling', '-wb', default=False, type=bool,
                    help='use weighted bundling for input layer when true, value hypervectors when false (default: True)')
parser.add_argument('--connected-mapping', '-cm', default=False, type=bool,
                    help='when true, uses fully connected hypervector mapping, when false, uses binding in convolutions (default: False)')
parser.add_argument('--multiple-prototype', '-mp', default=False, type=bool,
                    help='when true, uses multiple prototypes to classify each feature hypervector, when false, uses global position keys to map feature hypervector map to a single hypervector (default: False)')
parser.add_argument('--affine-bn', '-bn', default=True, type=bool,
                    help='use affine parameters for batch norm (default: True)')
parser.add_argument('--nhidden', '-nh', default=1, type=int,
                    help='number of ConvHDC hidden layers (default: 1)')
parser.add_argument('--nblocks', '-nbl', default=3, type=int,
                    help='number of blocks for ResConvHDC (default: 3)')
parser.add_argument('--n_value_vectors', '-nv', default=10, type=int,
                    help='the number of value hypervectors to use to encode values (when using value vectors) (default: 10)')
parser.add_argument('--batch-size', '-b', default=16, type=int,
                    help='batch size (default: 16)')
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', '-m', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '-wd', default=0, type=float,
                    help='weight decay (default: 0)')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--pretrained', '-prt', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', '-sd',
                    help='The directory to save trained models and log training metrics',
                    default=os.path.join('train_results', 'experiment'), type=str)
parser.add_argument('--save-every', '-se',
                    help='Saves model every number of epochs',
                    type=int, default=50)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # set up augmentations and load dataset
    if args.dataset == "MNIST":
        transform_test = torchvision.transforms.ToTensor()
        transform_train = torchvision.transforms.ToTensor()

        # load MNIST
        train_ds_full = MNIST(args.dataset_download_path, train=True, transform=transform_train, download=True)
        train_ds_full_noaugmentations = MNIST(args.dataset_download_path, train=True, transform=transform_test, download=True)
        test_ds = MNIST(args.dataset_download_path, train=False, transform=transform_test, download=True)
        labels = train_ds_full.targets.numpy()

    elif args.dataset == "FashionMNIST":
        transform_test = torchvision.transforms.ToTensor()
        transform_train = torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        ])
        
        # load FasionMNIST
        train_ds_full = FashionMNIST(args.dataset_download_path, train=True, transform=transform_train, download=True)
        train_ds_full_noaugmentations = FashionMNIST(args.dataset_download_path, train=True, transform=transform_test, download=True)
        test_ds = FashionMNIST(args.dataset_download_path, train=False, transform=transform_test, download=True)
        labels = train_ds_full.targets.numpy()

    elif args.dataset == "CIFAR10":
        normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                    std=[0.2470, 0.2435, 0.2616])
        
        # only normalize if using weighted bundling
        if args.weighted_bundling:
            transform_test = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            normalize
                            ])
            transform_train = torchvision.transforms.Compose([
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomCrop(32, 4),
                            torchvision.transforms.ToTensor(),
                            normalize
                            ])
        else: 
            transform_test = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            ])
            transform_train = torchvision.transforms.Compose([
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomCrop(32, 4),
                            torchvision.transforms.ToTensor(),
                            ])

        # load CIFAR10
        train_ds_full = CIFAR10(args.dataset_download_path, train=True, transform=transform_train, download=True)
        train_ds_full_noaugmentations = CIFAR10(args.dataset_download_path, train=True, transform=transform_test, download=True)
        test_ds = CIFAR10(args.dataset_download_path, train=False, transform=transform_test, download=True)
        labels = np.array(train_ds_full.targets)
    
    # split the training set into train and validation subsets using a stratified split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=42)
    train_idx, val_idx = next(stratified_split.split(np.zeros(len(labels)), labels))

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(train_ds_full_noaugmentations, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    img_shape = train_ds_full[0][0].shape
    n_classes = len(np.unique(labels))

    criterion = nn.CrossEntropyLoss()

    # set up model
    if args.resume:
        # load saved data
        saved = torch.load(args.resume, weights_only=False)

        # track accuracies
        best_epoch = saved['epoch'] - 1
        best_loss = saved['train_loss']
        best_validation = saved['valid_acc']
        best_test = saved['test_acc']
        start_epoch = saved['epoch']

        model = saved['model']

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
    else:
        # track accuracies
        best_epoch = 0
        best_loss = 0
        best_validation = 0
        best_test = 0
        start_epoch = 0

        # create model
        if args.model_architecture == 'ConvHDC':
            model = ConvHDC(
                D=args.dimension,
                n_classes=n_classes,
                in_channels=img_shape[0], 
                n_hidden=args.nhidden,
                image_shape=img_shape[1],
                affine_bn=args.affine_bn,
                weighted_bundling=args.weighted_bundling,
                fully_connected_mapping=args.connected_mapping,
                multiple_prototype=args.multiple_prototype)
        elif args.model_architecture == 'ResConvHDC':
            model = ResConvHDC(
                D=args.dimension,
                n_classes=n_classes,
                in_channels=img_shape[0],
                n_value_vectors=args.n_value_vectors,
                n_blocks=args.nblocks,
                image_shape=img_shape[1],
                affine_bn=args.affine_bn,
                weighted_bundling=args.weighted_bundling,
                fully_connected_mapping=args.connected_mapping,
                multiple_prototype=args.multiple_prototype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.to(device)

    # paths for logging metrics and saving model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_config_path = os.path.join(save_dir, 'train_config.txt')
    log_path = os.path.join(save_dir, 'training_log.txt')

    # save training configuration
    with open(train_config_path, 'w') as config_file: 
        config_file.write("Training Configuration:\n\n")
        print("Training Configuration:")
        for arg, value in vars(args).items():
            config_line = str(arg) + ": " + str(value)
            config_file.write(config_line + "\n")
            print(config_line)
    
    # train loop
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0

        # train
        model.train()
        for samples, labels in tqdm(train_loader, desc='Training (epoch ' + str(epoch + 1) + ')', leave=False):
            samples = samples.to(device)
            labels = labels.to(device)

            out = model(samples)
            loss = criterion(out, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # compute accuracy for the validation and test sets
        validation_total_correct = 0
        test_total_correct = 0
        model.eval()
        with torch.no_grad():
            for samples, labels in validation_loader:
                samples = samples.to(device)
                labels = labels.to(device)

                out = model(samples)
                predicted_classes = torch.argmax(out, dim=1)
                num_correct = (labels == predicted_classes).int().sum()

                validation_total_correct += num_correct.item()
        
            for samples, labels in test_loader:
                samples = samples.to(device)
                labels = labels.to(device)

                out = model(samples)
                predicted_classes = torch.argmax(out, dim=1)
                num_correct = (labels == predicted_classes).int().sum()

                test_total_correct += num_correct.item()

        # compute metrics
        train_loss = epoch_loss / len(train_loader)
        validation_accuracy = validation_total_correct / len(validation_loader.dataset)
        test_accuracy = test_total_correct / len(test_loader.dataset)

        epoch_metric_str = f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f} | Valid Acc: {validation_accuracy:.2%} | Test Acc {test_accuracy:.2%}"

        # log metrics
        mode = 'w' if epoch == start_epoch else 'a'
        with open(log_path, mode) as log_file: 
            if epoch == start_epoch:
                log_file.write("Training Metrics:\n")
            log_file.write(epoch_metric_str + "\n")
        
        # save best model
        if validation_accuracy > best_validation:
            best_epoch = epoch + 1
            best_test = test_accuracy
            best_validation = validation_accuracy
            best_loss = epoch_loss

            save_model_path = os.path.join(save_dir, 'best.pt')
            torch.save({
                'epoch': best_epoch,
                'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': best_loss,
                'valid_acc': best_validation,
                'test_acc': best_test,
                }, save_model_path)
            
            epoch_metric_str += " (new best, model saved)"
        
        # save model
        elif (epoch + 1) % args.save_every == 0:
            save_model_path = os.path.join(save_dir, 'epoch_' + f'{epoch+1:03d}' + '.pt')
            torch.save({
                'epoch': epoch + 1,
                'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_acc': validation_accuracy,
                'test_acc': test_accuracy,
                }, save_model_path)
            
            epoch_metric_str += " (model saved)"
        
        print(epoch_metric_str)
