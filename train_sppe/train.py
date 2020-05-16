# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import pandas as pd
import slackweb

from utils.dataset import animal
from opt import opt
from models.FastPose import createModel
from utils.eval import DataLogger, accuracy
from utils.img import flip_lr, shuffleLR

# from evaluation import prediction

WEB_HOOK_URL = (
    "https://hooks.slack.com/services/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"
)


def send_slack_notification(
    message, username="Alpha Pose", icon_emoji=":hourglass_flowing_sand:"
):
    """
   This script sends a message to slack notification.
   """
    slack = slackweb.Slack(url=WEB_HOOK_URL)
    slack.notify(text=message, username=username, icon_emoji=icon_emoji)


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps)

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar("Train/Loss", lossLogger.avg, opt.trainIters)
        writer.add_scalar("Train/Acc", accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            "loss: {loss:.8f} | acc: {acc:.2f}".format(
                loss=lossLogger.avg, acc=accLogger.avg * 100
            )
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip_lr(inps))
            flip_out = flip_lr(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar("Valid/Loss", lossLogger.avg, opt.valIters)
        writer.add_scalar("Valid/Acc", accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            "loss: {loss:.8f} | acc: {acc:.2f}".format(
                loss=lossLogger.avg, acc=accLogger.avg * 100
            )
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():

    # Model Initialize
    m = createModel().cuda()
    if opt.loadModel:
        print("Loading Model from {}".format(opt.loadModel))
        m.load_state_dict(torch.load(opt.loadModel))
        if not os.path.exists("exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("exp/{}".format(opt.dataset))
                os.mkdir("exp/{}/{}".format(opt.dataset, opt.expID))
    else:
        print("Create new model")
        if not os.path.exists("exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("exp/{}".format(opt.dataset))
                os.mkdir("exp/{}/{}".format(opt.dataset, opt.expID))

    criterion = torch.nn.MSELoss().cuda()

    if opt.optMethod == "rmsprop":
        optimizer = torch.optim.RMSprop(
            m.parameters(),
            lr=opt.LR,
            momentum=opt.momentum,
            weight_decay=opt.weightDecay,
        )
    elif opt.optMethod == "adam":
        optimizer = torch.optim.Adam(m.parameters(), lr=opt.LR)
    else:
        raise Exception

    writer = SummaryWriter(".tensorboard/{}/{}".format(opt.dataset, opt.expID))

    # for my own Dataset
    if opt.dataset == "animal":
        train_dataset = animal.AnimalDataset(train=True)
        val_dataset = animal.AnimalDataset(train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.trainBatch,
        shuffle=True,
        num_workers=opt.nThreads,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.validBatch,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True,
    )

    # Model Transfer
    m = torch.nn.DataParallel(m).cuda()

    logs = []
    # Start Training
    for i in range(opt.nEpochs):
        opt.epoch = i

        print("############# Starting Epoch {} #############".format(opt.epoch))
        train_loss, train_acc = train(train_loader, m, criterion, optimizer, writer)

        print(
            "Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}".format(
                idx=opt.epoch, loss=train_loss, acc=train_acc
            )
        )

        # Send train info to slack
        send_slack_notification(
            message="Train - {idx:d}epoch\n loss : {loss:.8f}\n acc : {acc:.4f}".format(
                idx=opt.epoch, loss=train_loss, acc=train_acc
            )
        )

        opt.acc = train_acc
        opt.loss = train_loss
        m_dev = m.module
        if i % opt.snapshot == 0:
            torch.save(
                m_dev.state_dict(),
                "exp/{}/{}/model_{}.pkl".format(opt.dataset, opt.expID, opt.epoch),
            )
            torch.save(
                opt, "exp/{}/{}/option.pkl".format(opt.dataset, opt.expID, opt.epoch)
            )
            torch.save(
                optimizer, "exp/{}/{}/optimizer.pkl".format(opt.dataset, opt.expID)
            )

        val_loss, val_acc = valid(val_loader, m, criterion, optimizer, writer)

        print(
            "Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}".format(
                idx=i, loss=val_loss, acc=val_acc
            )
        )

        # Send val info to slack
        send_slack_notification(
            message="Valid - {idx:d}epoch\n loss : {loss:.8f}\n acc : {acc:.4f}".format(
                idx=i, loss=val_loss, acc=val_acc
            )
        )

        # Store loss and acc
        log = {
            "epoch": i,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        logs.append(log)

        """
        if opt.dataset != 'mpii':
            with torch.no_grad():
                mAP, mAP5 = prediction(m)

            print('Prediction-{idx:d} epoch | mAP:{mAP:.3f} | mAP0.5:{mAP5:.3f}'.format(
                idx=i,
                mAP=mAP,
                mAP5=mAP5
            ))
        """
    writer.close()
    # Save logs
    df = pd.DataFrame(logs)
    df.to_csv("./log/loss.csv", index=False)


if __name__ == "__main__":
    main()
