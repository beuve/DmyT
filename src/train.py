import argparse
from enum import Enum
import os

from tqdm import tqdm
import math

import torch
import timm
from timm.data import resolve_data_config

from loss.losses import Losses
from loader import loader
from utils import get_feature_size


class LEARNING_STEP(Enum):
    VALIDATION = 1
    TRAINING = 0


def get_dummy_prediction(output_features, dummies, device):
    preds = torch.zeros((output_features.size(0), )).to(device)
    for i, feat in enumerate(output_features):
        max_dot_value = -math.inf
        for dum_nb, dum in enumerate(dummies):
            dot_value = torch.dot(feat, dum / dum.sum())
            if max_dot_value < dot_value:
                max_dot_value = dot_value
                preds[i] = dum_nb
    return preds


def one_epoch(model, data_loader, criterion, optimizer, device, step):

    epoch_loss = 0.0
    epoch_accuracy = 0.0

    batch_nb = len(data_loader)
    for (data, target) in tqdm(data_loader, total=batch_nb):

        data, target = data.to(device), target.to(device)

        if step == LEARNING_STEP.VALIDATION:
            model.eval()
            with torch.no_grad():
                output = model.forward(data)
                loss = criterion(output, target.argmax(dim=1))
                if criterion.name == 'DmyT':
                    output = get_dummy_prediction(output,
                                                  criterion._loss.dummies,
                                                  device)
                elif criterion.name == 'Triplet':
                    output = torch.zeros((target.size(0))).to(device)
                else:
                    output = output.argmax(dim=1)

        else:
            model.train()
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, target.argmax(dim=1))
            loss.backward()
            if criterion.name == 'DmyT':
                output = get_dummy_prediction(output, criterion._loss.dummies,
                                              device)
            elif criterion.name == 'Triplet':
                output = torch.zeros((target.size(0))).to(device)
            else:
                output = output.argmax(dim=1)
            optimizer.step()

        accuracy = (output == target.argmax(dim=1)).float().mean()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    global_acc = epoch_accuracy / len(data_loader)
    global_loss = epoch_loss / len(data_loader)

    return global_loss, global_acc


def fit(
    model,
    epochs,
    loss,
    optimizer,
    scheduler,
    device,
    train_loader,
    valid_loader,
    output,
) -> None:

    best_valid_acc = 0. if loss.name != 'Triplet' else math.inf

    for epoch in range(1, epochs + 1):

        print("=" * 20)
        print(f"EPOCH {epoch} TRAINING...")

        train_loss, train_acc = one_epoch(
            model,
            train_loader,
            loss,
            optimizer,
            device,
            LEARNING_STEP.TRAINING,
        )
        print(
            f"[TRAIN] EPOCH {epoch} - LOSS: {train_loss:2.4f}, ACCURACY:{train_acc:2.4f}"
        )

        valid_loss, valid_acc = 0, 0
        if valid_loader is not None:
            print("EPOCH " + str(epoch) + " - VALIDATING...")

            valid_loss, valid_acc = one_epoch(
                model,
                valid_loader,
                loss,
                optimizer,
                device,
                LEARNING_STEP.VALIDATION,
            )

            if (valid_acc > best_valid_acc and loss.name != 'Triplet') or (
                    valid_loss < best_valid_acc and loss.name == 'Triplet'):
                print('--- ACC has improved, saving model ---')
                torch.save(model.state_dict(), output)
                best_valid_acc = valid_acc if loss.name != 'Triplet' else valid_loss

            print(
                f"[VALID] LOSS: {valid_loss:2.4f}, ACCURACY:{valid_acc:2.4f}")

        scheduler.step()


def main(args):
    model_name = args.model
    learning_rate = args.lr
    epochs = args.epoch
    batch_size = args.batch
    output_file = args.output
    data_folder = args.dataset
    weights = args.label_weights
    labels = os.listdir(os.path.join(data_folder, 'train'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_size = get_feature_size(model_name, device)
    label_weights = torch.tensor(weights,
                                 device=device) if weights != [] else None
    loss = Losses.from_string(args.loss, device, len(labels), feature_size,
                              label_weights)
    if loss.name == 'CrossEntropy':
        model = timm.create_model(model_name,
                                  pretrained=True,
                                  num_classes=len(labels))
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model_config = resolve_data_config({}, model=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.5)

    print('=' * 10, 'CONFIG', '=' * 10)
    print('MODEL:       ', model_name)
    print('LR:          ', learning_rate)
    print('BATCH_SIZE:  ', batch_size)
    print('EPOCH:       ', epochs)
    print('OPTIMIZER:   ', optimizer)
    print('LOSS:        ', loss.name)
    print('SCHEDULER:   ', scheduler)
    print('OUTPUT:      ', output_file)
    print('WEIGHTS:     ', label_weights)
    print('LABELS:      ', labels)
    print('=' * 26)

    train_loader = loader(data_folder, model_config['input_size'][1],
                          batch_size, "train")
    valid_loader = loader(data_folder, model_config['input_size'][1],
                          batch_size, "valid")
    model.to(device)

    fit(
        model,
        epochs,
        loss,
        optimizer,
        scheduler,
        device,
        train_loader,
        valid_loader,
        output_file,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="xception",
        type=str,
        help="TIMM model name",
    )
    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=8,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "-l",
        "--loss",
        default='DmyT',
        type=str,
        help="Loss to be used for training: CrossEntropy or Triplet or DmyT",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        default=10,
        type=int,
        help="number of epochs",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/data",
        type=str,
        help="Repository containing the dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='test.pth',
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "-lw",
        "--label-weights",
        default=None,
        nargs='*',
        type=float,
        help="Weights corresponding to each label",
    )

    args = parser.parse_args()
    main(args)
