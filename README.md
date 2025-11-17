# Lightweight Speech Enhancement via Learnable Prior and Schrödinger Bridge GAN

Official PyTorch Implementation of "Lightweight Speech Enhancement via Learnable Prior and Schrödinger Bridge GAN" by Zengqiang Shang, Biao Liu, Mou Wang, Xin Liu, and Pengyuan Zhang.

This paper introduces a novel speech enhancement framework that integrates learnable prior with a Schrödinger Bridge Generative Adversarial Network. While conventional Schrödinger Bridge-based speech enhancement methods have shown promising results, they suffer from inefficient transport paths due to path crossings, high computational requirements, and degraded speech quality in resource-constrained scenarios. The proposed approach overcomes these limitations by synergistically combining learnable prior and adversarial modeling paradigms.

## Key Features

- **Learnable Prior Module**: Effectively captures fundamental speech characteristics and dynamically adjusts the initial distribution during training, minimizing path crossings and facilitating more efficient transport paths
- **Adversarial Training**: Incorporates adversarial training and multi-scale loss functions to enhance speech quality and naturalness
- **Computational Efficiency**: Maintains computational efficiency while delivering superior performance
- **Resource-Constrained Performance**: Demonstrates robust performance even in resource-constrained environments, with the nano-sized model (0.04M parameters) delivering competitive results
- **State-of-the-Art Results**: Outperforms state-of-the-art approaches across various metrics, including OVRL, SIG, BAK, and P808.MOS

# Installation

- Create a new virtual environment with `Python >= 3.8`.
- Install the required packages by running

```bash
pip install -r requirements.txt
```

**(IMPORTANT)** recommend `torch >= 2.2.0`

- (optional) if using W&B logging
  - Set up a [wandb.ai](https://wandb.ai/) account
  - Login in via `wandb login <your_api_key>`

# Datasets

We use `Voicebank+DEMAND` and `TIMIT+WHAM!` for training and testing.

**(IMPORTANT)** Please set the path of these two dataset directories by the `datasets` attribute in `config/default_dataset.yml`. You can also add the paths to your custom dataset directories.
Each dataset directories should contain the following structure:

```angular2html
- train
    - clean
    - noisy
- valid
    - clean
    - noisy
- test
    - clean
    - noisy
```

# Training

The training process involves two main components: the learnable prior module and the Schrödinger Bridge GAN framework.

To pre-train the discriminator, run

```bash
torchrun --nproc_per_node=4 train_discriminator.py --discriminator_backbone <choosed_backbone> --dataset <target_dataset> --gpus 0,1,2,3 --no_mean_inverting
```

To train the complete model with learnable prior and Schrödinger Bridge GAN, run

```bash
torchrun --nproc_per_node=4 train.py --dataset <target_dataset> --gpus 0,1,2,3 
```

Here are some key arguments you can modify:

- `--config_path`: Path to the YAML configuration file. The default is `config/default.yml`.
- `--dataset`: Dataset to use for training and evaluation.
- `--scorer_backbone`: Backbone architecture for the scorer.
- `--denoiser_backbone`: Backbone architecture for the denoiser.
- `--beta`: Value for the noise schedule 'Beta'.
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training.
- `--evaluate_batch_size`: Batch size for evaluation.
- `--patience`: Patience for early stopping mechanism.
- `--gpus`: Comma-separated list of GPU indices to use. The default is `0`.
- `--resume`: Resume from a previous checkpoint.
- `--wandb_log`: Whether to log to Weights & Biases.

For a full list of arguments, run `python train.py --help`.

# Inference

The lightweight nature of our model enables efficient inference even in resource-constrained environments. The nano-sized model (0.04M parameters) delivers competitive results while maintaining computational efficiency.

For inference, run

```bash
python enhancement.py --run_name <your_run_name> --dataset <target_dataset> --sampling_method hybrid --skip_type time_uniform --NFE 3 --calc_metrics --max_workers 8
```

Here are some key arguments you can modify:

- `--dataset`: Dataset to use for inference.
- `--run_name`: Name of the run to evaluate.
- `--sampling_method`: Sampling method for inference. Available options are `hybrid`, `ddim`.
- `--skip_type`: Type of skip sampling for inference. Available options are `time_uniform`, `logSNR`, `time_quadratic`.
- `--NFE`: The number of function evaluations for inference.
- `--calc_metrics`: Whether to calculate metrics.

For a full list of arguments, run `python enhancement.py --help`.

# Evaluating Metrics

Our method is evaluated using comprehensive metrics including OVRL, SIG, BAK, and P808.MOS, demonstrating superior performance compared to state-of-the-art approaches.

You can evaluate the metrics by running

```bash
python calculate_metrics.py --test_dir <path_to_your_testset> --enhanced_dir <path_to_your_enhanced_audios> --suffix .wav
```

Here are some key arguments you can modify:

- `--test_dir`: Path to the directory containing the testset.
- `--enhanced_dir`: Path to the directory containing the enhanced audios.
- `--suffix`: Suffix of the audio files.
- 

For a full list of arguments, run `python calculate_metrics.py --help`.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{shang2025lightweight,
  title={Lightweight Speech Enhancement via Learnable Prior and Schr{\"o}dinger Bridge GAN},
  author={Shang, Zengqiang and Liu, Biao and Wang, Mou and Liu, Xin and Zhang, Pengyuan},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments

This work was supported by the Laboratory of Speech and Intelligent Information Processing, Institute of Acoustics, CAS, China, and OPPO Hardware Engineering System.

## Reference Implementation

This implementation is based on the reference code from:

[MISB: Generative Speech Enhancement using Mean-Inverting Diffusion Schrödinger Bridge](https://github.com/ishine/MISB)
