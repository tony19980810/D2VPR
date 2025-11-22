
import os
import torch

import D2VPR

import parser
import logging
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws

import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
  
    ######################################### SETUP #########################################
    places = ['sped', 'msls_val','pitts250k', 'amstertime','pitts30k', 'nordland']

    for place in places:
        args = parser.parse_arguments_D2VPR()
        start_time = datetime.now()
        args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
        commons.setup_logging(args.save_dir)
        commons.make_deterministic(args.seed)

        args.features_dim = 14*768
        logging.info(f"Arguments: {args}")
        logging.info(f"The outputs are being saved in {args.save_dir}")

        ######################################### MODEL #########################################


        model = D2VPR.D2VPR()
        model = model.to(args.device)
       


        if place == 'sped':
            args.infer_batch_size = 3
            model.load_state_dict(
                torch.load('./distillation.pth')['model_state_dict'])
        if place == 'amstertime':
            args.infer_batch_size = 8
            model.load_state_dict(
                torch.load('./distillation.pth')['model_state_dict'])
        if place == 'pitts250k':
            args.infer_batch_size = 8
            model.load_state_dict(
                torch.load('./distillation.pth')['model_state_dict'])
        if place == 'pitts30k':
            args.infer_batch_size = 8
            model.load_state_dict(
                torch.load('./distillation.pth')['model_state_dict'])
        if place == 'msls_val':
            args.infer_batch_size = 128
            model.load_state_dict(
                torch.load('./finetune.pth')['model_state_dict'])
        if place == 'nordland':
            args.infer_batch_size = 128
            model.load_state_dict(
                torch.load('./finetune.pth')['model_state_dict'])

        args.pca_dim=4096
        if args.pca_dim is None:
            pca = None
        else:
            full_features_dim = args.features_dim
            args.features_dim = args.pca_dim
            pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

        ######################################### DATASETS #########################################
        test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, place, "test")
        logging.info(f"Test set: {test_ds}")

        ######################################### TEST on TEST SET #########################################
        recalls, recalls_str = test.test(args, test_ds, model, args.test_method, pca)
        logging.info(f"Recalls on {test_ds}: {recalls_str}")

        logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
