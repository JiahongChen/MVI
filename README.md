# Mutual variational inference: An indirect variational inference approach for unsupervised domain adaptation
This is the Python+PyTorch code to reproduce the results of domain adaptation in image classification in paper ['Mutual variational inference: An indirect variational inference approach for unsupervised domain adaptation'](https://ieeexplore.ieee.org/abstract/document/9529096/).

# Requirements
* Platform : Linux 
* Computing Environment:
  * CUDA 10.1 
  * PyTorch
* Packages: ```pandas, numpy, scipy, argparse, tqdm```.
* Hardware : Nvidia GPU

# Run the code
* Download ResNet-50 pretrained model and place it under ```model/```.
* Download necessary DA datasets and place it under ```data/```.
* Run bash file ```batchrun.sh```


# Citation
Please cite our paper if you found it usefull.
```
@article{chen2021mutual,
  title={Mutual variational inference: An indirect variational inference approach for unsupervised domain adaptation},
  author={Chen, Jiahong and Wang, Jing and de Silva, Clarence W},
  journal={IEEE Transactions on Cybernetics},
  year={2022},
  volume={52},
  number={11},
  pages={11491-11503},
  doi={10.1109/TCYB.2021.3107292}
}
```
