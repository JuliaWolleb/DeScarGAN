# DeScarGAN: Disease-Specific Anomaly Detection with Weak Supervision
This is the official Pytorch implementation of the paper DeScarGAN.


![output](Result_Chexpert.png)


Datasets
-------------------
For the generation of the synthetic dataset, one has to run the  script “create_synthetic_dataset”. A new folder called "warp-set" will be created and the generated images of both the diseased and healthy subjects will be stored in there.

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

Citation
--------------------
If you use this code, please cite @article{wolleb_descargan:,
	title = {{DeScarGAN}: {Disease-Specific} {Anomaly} {Detection} with {Weak} {Supervision}},
	author = {Wolleb, Julia and Sandk\"uhler, Robin and Cattin, Philippe C.},
	journal = {arXiv:},
	year = {2020}
}
