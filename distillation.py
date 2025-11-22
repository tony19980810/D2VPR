import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader


import network
import parser
import D2VPR

torch.backends.cudnn.benchmark = True  # Provides a speedup

import util
import test

import commons
import datasets_ws

from loss import loss_function, distillation_loss
from dataloaders.GSVCities import get_GSVCities

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':




    #### Initial setup: parser, logging...
    args = parser.parse_arguments_D2VPR()
    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    #### Creation of Datasets
    logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

    test_ds_pitts30k = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 'pitts30k', "test")
    test_ds_msls = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 'msls_val', "test")
    test_ds_sped = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 'sped', "test")
    args.features_dim = 14*768
    #### Initialize model
    model = D2VPR.D2VPR(pretrained_foundation = True, foundation_model_path = args.foundation_model_path,cross_image_encoder=False)

    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    ## initialize Adapter
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'conv' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)


    if args.teacher == True:
        teacher = network.CricaVPRNet().eval().to(args.device)
        if args.resume is not None:
            logging.info(f"Resuming model from {args.resume}")
            teacher = util.resume_model(args, teacher)

    #### Setup Optimizer and Loss
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    best_r5 = start_epoch_num = not_improved_num = 0
    logging.info(f"Output dimension of the model is {args.features_dim}")

    #### Getting GSVCities
    train_dataset = get_GSVCities()

    train_loader_config = {
        'batch_size': args.train_batch_size,
        'num_workers': args.num_workers,
        'drop_last': False,
        'pin_memory': True,
       'shuffle': False}

    #### Training loop
    ds = DataLoader(dataset=train_dataset, **train_loader_config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0,1), dtype=np.float32)

        model = model.train()
        epoch_losses=[]
        for images, place_id in tqdm(ds):
            BS, N, ch, h, w = images.shape

            # reshape places and labels
            images = images.view(BS*N, ch, h, w)
            #
            labels = place_id.view(-1)

            descriptors= model(images.to(args.device))

            if args.teacher:
                teacher_descriptors = teacher(images.to(args.device))
                loss = distillation_loss(descriptors, teacher_descriptors)
            else:
                loss = loss_function (descriptors, labels)



            del descriptors

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss

        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        # Compute recalls on validation set
        recalls_pitts30k, recalls_str_pitts30k = test.test(args, test_ds_pitts30k, model)
        logging.info(f"Recalls on val set {test_ds_pitts30k}: {recalls_str_pitts30k}")

        recalls_msls, recalls_str_msls = test.test(args, test_ds_msls, model, args.test_method)
        logging.info(f"Recalls on test set {test_ds_msls}: {recalls_str_msls}")

        recalls_sped, recalls_str_sped = test.test(args, test_ds_sped, model, args.test_method)
        logging.info(f"Recalls on test set {test_ds_sped}: {recalls_str_sped}")

        is_best = recalls_msls[1] > best_r5

        # Save checkpoint, which contains all training parameters
        util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls_msls, "best_r5": best_r5,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth")





        # If recall@5 did not improve for "many" epochs, stop training
        if is_best:
            logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {(recalls_msls[1]):.1f}")
            best_r5 = (recalls_msls[1])
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {(recalls_msls[1]):.1f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break

    logging.info(f"Best R@5: {best_r5:.1f}")
    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

