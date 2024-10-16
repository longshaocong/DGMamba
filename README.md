# DGMamba: Domain Generalization via Generalized State Space Model

Welcome to the repository for our paper: "DGMamba: Domain Generalization via Generalized State Space Model."

## Installation
### Environments

Environment details used for the main experiments. 
```
Environment:
	Python: 3.7.13
	PyTorch: 1.12.1
	Torchvision: 0.13.1
	CUDA: 10.2
	CUDNN: 7605
	NumPy: 1.21.5
	PIL: 9.5.0
	selective_scan: 0.0.2
```

### Dependencies

```sh
pip install -r requirements.txt
```


## Usage

### Checkpoints
Below, we provide the checkpoints of DGMamba for PACS.

[DownLoad Link of Google Drive](https://drive.google.com/drive/folders/1xYPaT2RIutpqQnZ_jY9KXKM0trQc_zhs?usp=drive_link)

### Training

#### Training on single node

You can use the following training command to train DGMamba.
We provide the sample on PACS with 'Art painting' as the target domain.

```bash
CUDA_VISIBLE_DEVICES='0' CUDA_LAUNCH_BLOCKING=1 python -u -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 11773 main.py --cfg ./configs/vssm_tiny_224_0220.yaml --data-path your_data_path --lr 3e-4\
		--algorithm DGMamba --output ./train_output --dataset PACS --test_envs 0  --pretrained pretrained_file
```


## Acknowledgements
This project is based on VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)). We thank their authors for making the source code publically available.


## Citation

If you find DGMamba useful in your research, please consider citing:
```bibtex
@inproceedings{long2024dgmamba,
  title={Dgmamba: Domain generalization via generalized state space model},
  author={Long, Shaocong and Zhou, Qianyu and Li, Xiangtai and Lu, Xuequan and Ying, Chenhao and Luo, Yuan and Ma, Lizhuang and Yan, Shuicheng},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2024}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.