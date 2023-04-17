# Partial Feature Distillation Network

**Winner** of **Overall evaluation** and **4th** of **Runtime** in the [NTIRE 2023 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2023/).

Paper and training codes will come soon!

| <sub> Model </sub> | <sub> Runtime[ms] </sub> | <sub> Params[M] </sub> | <sub> Flops[G] </sub> |  <sub> Acts[M] </sub> | <sub> GPU Mem[M] </sub> |
|  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  |
|  RFDN  | 35.54  |  0.433  | 27.10  |  112.03  | 788.13  |
|  PFDN  | 20.49  |  0.272  | 16.76  |  65.10  | 296.45  |
## How to test the baseline model?

1. `git clone https://github.com/icandle/PFDN.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 8
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.

## How to download the results?

1. `pip install gdown`
2. DIV2K_valid Results [[Google Drive Link](https://drive.google.com/file/d/1sgCbdmJAU_NoLydii88qi0xlg23vTVdB/view?usp=share_link)]
    ```bash
    gdown 'https://drive.google.com/uc?id=1sgCbdmJAU_NoLydii88qi0xlg23vTVdB'
    ```
    LSDIR_DIV2K_test Results [[Google Drive Link](https://drive.google.com/file/d/1CgPkAi0TcCVB85_T7HpXcEcPtfZfQWtl/view?usp=share_link)]
    ```bash
    gdown 'https://drive.google.com/uc?id=1CgPkAi0TcCVB85_T7HpXcEcPtfZfQWtl'
    ```

## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RFDN import RFDN
    model = RFDN()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
