# Expert-LaSTS: Expert-Knowledge Guided Latent Space for Traffic Scenarios

This repository provides the architecture and training as presented in "Expert-LaSTS: Expert-Knowledge Guided Latent Space for Traffic Scenarios".

## Setup and Train the Network
1. Clone the repository.
2. Download the data from https://faubox.rrze.uni-erlangen.de/getlink/fiXiigoqRjsuHSuCQeNXzx4t/Expert-LaSTS_data.zip and unzip it into a folder `\data`, rooted in the project directory. 
3. Run `python main.py` to train the model. Adjust parameters as required in `main.py`.

## Sources
This repository is based on the former version https://github.com/JWTHI/ViTAL-SCENE. The implementation is changed to handle dynamic infroamtion aside of the formerly used infrastructure information. The Vision-Transformer implementation is realized through `vit_pytorch.py`, provided in [vit-pytorch] (https://github.com/lucidrains/vit-pytorch).


## Citation
If you are using this repository, please cite the work
```
@InProceedings{Wurst2022a,
  author    = {Jonas {Wurst} and Lakshman {Balasubramanian} and Michael {Botsch} and Wolfgang {Utschick}},
  title     = {Expert-LaSTS: Expert-Knowledge Guided Latent Space for Traffic Scenarios},
  booktitle = {2022 IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2022},
}
```
