#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：DRG_parsing
@File ：run.py
@Author ：xiao zhang
@Date ：2022/11/14 12:27
'''

import argparse
import os
import sys
sys.path.append(".")

from model import get_dataloader, Generator

# path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", required=False, type=str, default="en",
                        help="language in [en, nl, de ,it]")
    parser.add_argument("-ip", "--if_pre", required=False, type=bool,
                        default=True,
                        help="if pre-train or not")
    parser.add_argument("-pt", "--pretrain", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/en/train/adjective_files/gold_adjectives_x2.sbn"),
                        help="pre-train(fine-tuning) sets")
    parser.add_argument("-t", "--train", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/en/train/original_dataset_files/gold.sbn"),
                        help="train sets")
    parser.add_argument("-d", "--dev", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/en/dev/standard.sbn"),
                        help="dev sets")
    parser.add_argument("-e", "--test", required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/en/test/standard.sbn"),
                        help="standard test sets")
    parser.add_argument("-c", "--challenge", nargs='*', required=False, type=str,
                        default=os.path.join(path, "data/pmb-5.0.0/seq2seq/en/test/long.sbn"),
                        help="challenge sets")
    parser.add_argument("-s", "--save", required=False, type=str,
                        default=os.path.join(path, "src/model/byT5/adjective_results"),
                        help="path to save the result")
    parser.add_argument("-epoch", "--epoch", required=False, type=int,
                        default=16)
    parser.add_argument("-lr", "--learning_rate", required=False, type=float,
                        default=1e-04)
    parser.add_argument("-ms", "--model_save", required=False, type=str,
                        default=os.path.join(path, "models/adjective_byT5_seq2seq/{lang}"))
    args = parser.parse_args()
    return args


def main():
    args = create_arg_parser()

    # train process
    lang = args.lang

    # train loader
    train_dataloader_pre = get_dataloader(args.pretrain)
    train_dataloader = get_dataloader(args.train)

    if args.if_pre:
        train_dataloader_pre = get_dataloader(args.pretrain)

    # test loader
    test_dataloader = get_dataloader(args.test)
    dev_dataloader = get_dataloader(args.dev)

    # save path
    save_path = args.save

    # hyperparameters
    epoch = args.epoch
    lr = args.learning_rate

    # train
    bart_classifier = Generator(lang)
    if args.if_pre: # if pretrain or not
        bart_classifier.train(train_dataloader_pre, dev_dataloader, lr=lr, epoch_number=3)
    bart_classifier.train(train_dataloader, dev_dataloader, lr=lr, epoch_number=epoch)

    # standard test
    bart_classifier.evaluate(test_dataloader, os.path.join(save_path,"gold/byT5_adj_aug_pars.txt"))

    # challenge test
    for i in range(len(args.challenge)):
        cha_path = args.challenge[i]
        cha_dataloader = get_dataloader(cha_path)
        bart_classifier.evaluate(cha_dataloader, os.path.join(save_path, f"challenge/challenge{i}.sbn"))

    bart_classifier.model.save_pretrained(args.model_save)


if __name__ == '__main__':
    main()