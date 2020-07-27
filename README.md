# DeScarGAN
This is the official Pytorch implementation of the paper DeScarGAN.

For the generation of the synthetic dataset, one has to run the  script “create_synthetic_dataset”. The generated images will be saved in the folder “data”.

If one wants to use the Chexpert dataset, it is available for download on this page: https://stanfordmlgroup.github.io/competitions/chexpert/.
The data has to have the following structure:


    • Train
        ◦ Healthy control
        ◦ Images showing pleural effusion
    • Validate
        ◦ Healthy control
        ◦ Images showing pleural effusion
    • Test
        ◦ Healthy control
        ◦ Images showing pleural effusion

To run the code, one needs to run “main.py”. There are the following options:

`--dataset`:   One has to determine which dataset has to be used (either “Chexpert” or “Synthetic”). 

`--dataset-path`:  The path to the data folders.

`--mode`:  One can choose between the mode “train” or “test”. When training, the networks are saved in the folder “./save_nets”. During test-time, those networks are loaded. Plots showing the results  can be seen using visdom (see https://github.com/facebookresearch/visdom for documentation).

If you use this code, please cite "\arxiv"
