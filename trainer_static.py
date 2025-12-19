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

from .models.StaticHDC import StaticHDC

parser = argparse.ArgumentParser()
parser.add_argument('--encoding-method', '-em', default='orthogonal',
                    choices=['orthogonal', 'linear', 'local_linear'], help='hypervector generation method (default: orthogonal)')
parser.add_argument('--num-in-split', '-s', default=5, type=int,
                    help='number of hypervectors in each split (only for local linear encoding) (default: 5)')
parser.add_argument('--dataset', '-ds', default='MNIST',
                    choices=['MNIST', 'FashionMNIST', 'CIFAR10'], help='dataset for training (default: MNIST)')
parser.add_argument('--n_value_vectors', '-nv', default=10, type=int,
                    help='the number of value hypervectors to use to encode values (when using value vectors) (default: 10)')
parser.add_argument('--dataset-download-path', '-ddp', default='../data',
                    help='path to download the data to (default: ../data)')
parser.add_argument('--epochs', '-ep', default=200, type=int,
                    help='number of epochs to train for (default: 200)')
parser.add_argument('--dimension', '-D', default=10000, type=int,
                    help='hypervector dimension (default: 10000)')
parser.add_argument('--batch-size', '-b', default=16, type=int,
                    help='batch size (default: 16)')
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='path to saved model to use for resumed training (default: none)')
parser.add_argument('--evaluate', '-e', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--pretrained', '-prt', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', '-sd',
                    help='The directory to save trained models and log training metrics',
                    default=os.path.join('train_results', 'experiment_static'), type=str)
parser.add_argument('--save-every', '-se',
                    help='Saves model every number of epochs',
                    type=int, default=50)


def compute_accuracy(model, encodings, labels):
    correct = 0
    for i in tqdm(range(len(encodings)), desc=f"Testing", leave=False):
        encoding = encodings[i].float()
        label = labels[i].long()

        prediction = model.classify(encoding)
        if prediction == label:
            correct += 1
    accuracy = correct / len(encodings)
    return accuracy



if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform_test = torchvision.transforms.ToTensor()
    transform_train = torchvision.transforms.ToTensor()

    # set up augmentations and load dataset
    if args.dataset == "MNIST":
        # load MNIST
        train_ds_full = MNIST(args.dataset_download_path, train=True, transform=transform_train, download=True)
        train_ds_full_noaugmentations = MNIST(args.dataset_download_path, train=True, transform=transform_test, download=True)
        test_ds = MNIST(args.dataset_download_path, train=False, transform=transform_test, download=True)
        labels = train_ds_full.targets.numpy()

    elif args.dataset == "FashionMNIST":
        # load FasionMNIST
        train_ds_full = FashionMNIST(args.dataset_download_path, train=True, transform=transform_train, download=True)
        train_ds_full_noaugmentations = FashionMNIST(args.dataset_download_path, train=True, transform=transform_test, download=True)
        test_ds = FashionMNIST(args.dataset_download_path, train=False, transform=transform_test, download=True)
        labels = train_ds_full.targets.numpy()

    elif args.dataset == "CIFAR10":
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

        # set up model
    if args.resume:
        # load saved data
        saved = torch.load(args.resume, weights_only=False)

        # track accuracies
        best_epoch = saved['epoch'] - 1
        best_train = saved['train_acc']
        best_validation = saved['valid_acc']
        best_test = saved['test_acc']
        start_epoch = saved['epoch']

        model = saved['model']
    else:
        # track accuracies
        best_epoch = 0
        best_train = 0
        best_validation = 0
        best_test = 0
        start_epoch = 0

        # create model
        model = StaticHDC(
            D=args.dimension,
            n_classes=n_classes,
            in_channels=img_shape[0], 
            n_value_vectors=args.n_value_vectors,
            image_shape=img_shape[1],
            mode=args.encoding_method,
            S=args.num_in_split
        )
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

    with torch.no_grad():
        # load train encodings in cpu memory
        train_encodings = torch.empty((len(train_ds), args.dimension), dtype=torch.int8, device="cpu")
        train_labels = torch.empty(len(train_ds), device="cpu")

        # build class prototypes
        i = 0
        for samples, labels in tqdm(train_loader, desc="Creating class prototypes and loading training encodings"):
            samples = samples.to(device)
            labels = labels.to(device)

            encodings = model.encode(samples)

            if not args.resume:
                model.classifier.add(encodings.to(device), labels)

            encodings = encodings.cpu().to(torch.int8)

            train_encodings[i:i+args.batch_size] = encodings
            train_labels[i:i+args.batch_size] = labels.cpu()

            i += args.batch_size
        
        # store validation encodings in cpu memory
        valid_encodings = torch.empty((len(val_ds), args.dimension), dtype=torch.int8, device="cpu")
        valid_labels = torch.empty(len(val_ds), device="cpu")

        # get validation encodings
        i = 0
        for samples, labels in tqdm(validation_loader, desc="Loading validation encodings"):
            samples = samples.to(device)
            labels = labels.to(device)

            encodings = model.encode(samples)
            encodings = encodings.cpu().to(torch.int8)

            valid_encodings[i:i+args.batch_size] = encodings
            valid_labels[i:i+args.batch_size] = labels.cpu()

            i += args.batch_size

        # store test encodings in cpu memory
        test_encodings = torch.empty((len(test_ds), args.dimension), dtype=torch.int8, device="cpu")
        test_labels = torch.empty(len(test_ds), device="cpu")

        # get test encodings
        i = 0
        for samples, labels in tqdm(test_loader, desc="Loading test encodings"):
            samples = samples.to(device)
            labels = labels.to(device)

            encodings = model.encode(samples)
            encodings = encodings.cpu().to(torch.int8)

            test_encodings[i:i+args.batch_size] = encodings
            test_labels[i:i+args.batch_size] = labels.cpu()

            i += args.batch_size

        # retraining loop
        model.to(torch.device('cpu'))
        for epoch in range(start_epoch, args.epochs):
            
            # retrain
            for i in tqdm(range(len(train_ds)), desc=f"Retraining (epoch {epoch})", leave=False):
                encoding = train_encodings[i].float()
                label = train_labels[i].long()

                prediction = model.classify(encoding)

                if label != prediction:
                    encoding = encoding.unsqueeze(0)
                    prediction = prediction.unsqueeze(0)
                    label = label.unsqueeze(0)

                    model.classifier.add(-encoding, prediction)
                    model.classifier.add(encoding, label)
            
            # compute accuracies for each set
            train_acc = compute_accuracy(model, train_encodings, train_labels)
            valid_acc = compute_accuracy(model, valid_encodings, valid_labels)
            test_acc = compute_accuracy(model, test_encodings, test_labels)

            epoch_metric_str = f"Epoch {epoch+1:03d}: Train Acc: {train_acc:.2%} | Valid Acc: {valid_acc:.2%} | Test Acc {test_acc:.2%}"

            # log metrics
            mode = 'w' if epoch == start_epoch else 'a'
            with open(log_path, mode) as log_file: 
                if epoch == start_epoch:
                    log_file.write("Training Metrics:\n")
                log_file.write(epoch_metric_str + "\n")

            # save best model
            if valid_acc > best_validation:
                best_epoch = epoch + 1
                best_test = test_acc
                best_validation = valid_acc
                best_train = train_acc

                save_model_path = os.path.join(save_dir, 'best.pt')
                torch.save({
                    'epoch': best_epoch,
                    'model': model,
                    'train_acc': best_train,
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
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'test_acc': test_acc,
                    }, save_model_path)
                
                epoch_metric_str += " (model saved)"
            
            print(epoch_metric_str)
        
        train_end_str = f"Best Train Acc: {best_train:.2%} | Best Valid Acc: {best_validation:.2%} | Best Test Acc {best_test:.2%}"
        with open(log_path, mode) as log_file: 
            log_file.write("\nEnd of training\n")
            log_file.write(train_end_str)