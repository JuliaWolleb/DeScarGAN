# DeScarGAN: Disease-Specific Anomaly Detection with Weak Supervision
This is the official Pytorch implementation of the paper DeScarGAN.


![output](Result_Chexpert.png)


Datasets
-------------------
For the generation of the synthetic dataset, one has to run the  script “create_synthetic_dataset”. A new folder called "warp-set" will be created and the generated images of both the diseased and healthy subjects will be stored in there seperately.

If one wants to use the Chexpert dataset, it is available for download on this page: https://stanfordmlgroup.github.io/competitions/chexpert/.
The data needs to be structured as follows:


    • Train
        ◦ Healthy control
        ◦ Images showing pleural effusion
    • Validate
        ◦ Healthy control
        ◦ Images showing pleural effusion
    • Test
        ◦ Healthy control
        ◦ Images showing pleural effusion

For the script “main.py”,There are the following options:

`--dataset`:   One has to determine which dataset has to be used (either “Chexpert” or “Synthetic”). 

`--dataset-path`:  The path to the data folders.

`--mode`:  One can choose between the mode “train” or “test”. When training, the networks are saved in the folder “./save_nets”. During test-time, the saved models are loaded. Plots showing the convergence of the loss functions and results are visualized using visdom (see https://github.com/facebookresearch/visdom for documentation).

Citation
--------------------
If you use this code, please cite 
@misc{wolleb2020descargan,
    title={DeScarGAN: Disease-Specific Anomaly Detection with Weak Supervision},
    author={Julia Wolleb and Robin Sandkühler and Philippe C. Cattin},
    year={2020},
    eprint={2007.14118},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
}
