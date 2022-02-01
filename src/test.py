import argparse
from tqdm import tqdm
import numpy as np
import math
import os

import timm
from timm.data import resolve_data_config
import torch
from sklearn.svm import SVC

from loss.losses import Losses
from loader import loader
from utils import get_feature_size


def get_output(model, device, loader, num=0):
    outputs = None
    labels = None
    batch_nb = len(loader)
    for i, (data, target) in tqdm(enumerate(loader), total=min(batch_nb, num)):
        data, target = data.to(device), target.to(device)

        if num > 0 and i >= num:
            break

        with torch.no_grad():
            output = model.forward(data)
            if outputs == None:
                outputs = output
                labels = target
            else:
                outputs = torch.cat((outputs, output), 0)
                labels = torch.cat((labels, target), 0)

    return outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()


def get_svm_predictions(feats, data_folder, model, device, input_size):
    svc = SVC(probability=True)
    train_loader = loader(data_folder, input_size, split='train')
    train_feats, train_labels = get_output(model,
                                           device,
                                           train_loader,
                                           num=5000)
    svc.fit(train_feats, train_labels.argmax(axis=1))
    raw_predicts = svc.predict_proba(feats)
    digits_predicts = svc.predict(feats)
    return digits_predicts


def get_dummy_predictions(feats, dummies):
    preds = np.zeros((feats.shape[0], ))
    for i, feat in enumerate(feats):
        max_dot_value = -math.inf
        for dum_nb, dum in enumerate(dummies):
            dot_value = np.dot(feat, dum[0][0])
            if max_dot_value < dot_value:
                max_dot_value = dot_value
                preds[i] = dum_nb
    return preds


def get_prediction(model, device, data_folder, loss, input_size):
    test_loader = loader(data_folder, input_size, split='test')
    test_feats, test_labels = get_output(model, device, test_loader)
    if loss.name == 'CrossEntropy':
        test_preds = test_feats.argmax(axis=1)
    elif loss.name == 'Triplet':
        test_preds = get_svm_predictions(test_feats, data_folder, model,
                                         device, input_size)
    else:
        test_preds = get_dummy_predictions(
            test_feats,
            list(
                map(lambda e: (e[0].detach().cpu().numpy(), e[1]),
                    loss.value.dummies)))
    return test_preds, test_labels


def accuracy(preds, labels):
    acc = (preds == labels.argmax(axis=1)).astype(float).mean()

    print(f'ACCURACY:            {100* acc:2.4f}')


def balanced_accuracy(outputs, gt, nb_class):
    accs = 0
    for i in range(nb_class):
        acc = ((outputs == i) * (gt.argmax(axis=1) == i)).astype(float).sum()
        nb = (gt.argmax(axis=1) == i).astype(float).sum()
        accs += acc / nb

    print(f'BALANCED ACCURACY:   {100* accs / nb_class:2.4f}')
    return


def separated_accuracy(outputs, gt, classes):
    print(f'TPR PER CLASS: ')
    for i in range(len(classes)):
        acc = ((outputs == i) * (gt.argmax(axis=1) == i)).astype(float).sum()
        nb = (gt.argmax(axis=1) == i).astype(float).sum()
        print(f'  {classes[i]} - {100* acc / nb:2.4f}')
    return


def main(args):
    model_name = args.model
    weights = args.weights
    data_folder = args.dataset
    labels = os.listdir(os.path.join(data_folder, 'test'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_size = get_feature_size(model_name, device)
    loss = Losses.from_string(args.loss, device, len(labels), feature_size)
    if loss.name == 'CrossEntropy':
        model = timm.create_model(model_name,
                                  pretrained=True,
                                  num_classes=len(labels))
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model_config = resolve_data_config({}, model=model)
    if args.weights != None:
        model.load_state_dict(torch.load(weights))
    model.to(device)
    model.eval()
    print('=' * 10, 'CONFIG', '=' * 10)
    print('MODEL:              ', model_name)
    print('WEIGHTS:            ', weights)
    print('=' * 28)
    outputs, gt = get_prediction(model, device, data_folder, loss,
                                 model_config['input_size'][1])
    print('=' * 10, 'RESULT', '=' * 10)
    accuracy(outputs, gt)
    balanced_accuracy(outputs, gt, nb_class=len(labels))
    separated_accuracy(outputs, gt, labels)
    print('=' * 28)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="xception",
        type=str,
        help="TIMM Model",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="path to model weights. Use imagenet from TIMM if unspecified.",
    )
    parser.add_argument(
        "-l",
        "--loss",
        default="DmyT",
        type=str,
        help="Loss to be used for training: CrossEntropy or Triplet or DmyT",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="/data",
        type=str,
        help="Repository containing the dataset",
    )

    args = parser.parse_args()
    main(args)
