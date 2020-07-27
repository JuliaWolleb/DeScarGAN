import os
import argparse

default_dataset = './dataset/warp_set/'

def main(config):
    dataset = config.dataset
    mode=config.mode
    print(mode)
    if mode=='train':

        from model.DeScarGAN import Solver

        solver = Solver(config.dataset_path, config.dataset)
        solver.train()
    else:
        if dataset == 'Synthetic':
            from Evaluation.Evaluation_Synthetic_Dataset import Solver
        else:
            from Evaluation.Evaluation_Chexpert import Solver

        solver = Solver(config.dataset_path, config.choose_net)
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    #     # Training configuration.
    parser.add_argument('--dataset_path', type=str,
                      default='/home/juliawolleb/PycharmProjects/Python_Tutorials/Reversible/Chexpert/2classes_effusion')
                    #    default='/home/juliawolleb/PycharmProjects/Python_Tutorials/warp/warp_set')
    parser.add_argument('--dataset', type=str, default='Chexpert')
    parser.add_argument('--mode',type=str, default='train')
    parser.add_argument('--choose_net',type=str, default='./save_nets')

    #
    #
    #
    config = parser.parse_args()
    print(config)
    main(config)
#
