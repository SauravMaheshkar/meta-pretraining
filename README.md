# [Meta-learning to Improve Pre-training, NeurIPS 2021 Poster](https://openreview.net/forum?id=Wiq6Mg8btwT)

This repository contains the code to reproduce the SimCLR experiments discussed in the paper and is a fork of the [original repository](https://github.com/aniruddhraghu/meta-pretraining) instrumented using [Weights and Biases](https://wandb.ai/site).


# Citation

```bibtex
@inproceedings{
    raghu2021metalearning,
    title={Meta-learning to Improve Pre-training},
    author={Aniruddh Raghu and Jonathan Peter Lorraine and Simon        Kornblith and Matthew B.A. McDermott and David Duvenaud},
    booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
    year={2021},
    url={https://openreview.net/forum?id=Wiq6Mg8btwT}
}
```

# 📝 Instruction
## 🏠 Setting up the environment

1. Create and activate a Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

2. Download the dataset using the provided bash file
```bash
./download_dataset.sh
```

Now let's run some experiments 😁😁😁😁.

## 💪🏻 PreTraining + FineTuning | PT + FT

To train a SimCLR model with default augmentations (doesn't train the augmentations)

```bash
python3 train.py --warmup_epochs 100 --epochs 50 --teacherarch warpexmag --studentarch resnet34 --seed <SEED>
```

**NOTE:** The warmup epochs being greater than the number of epochs means the augmentations are not optimized.

---
 
To train a SimCLR model and optimize augmentations, with `N` MetaFT examples, run:

```bash
python3 train.py --warmup_epochs 1 --epochs 50 --teacherarch warpexmag --seed SEED --ex N
```

## 📈 Evaluation

To fine-tune a pre-trained model on `NFT` fine-tuning examples (FT dataset has `N` data points), with FT seed `<RUNSEED>` and dataset seed (i.e., PT seed) `<SEED>`. For the partial FT access setting,  `NFT` is more than `N` from the PT :

```bash
python3 eval.py --training_epochs 50 --transfer_eval --runseed RUNSEED --seed SEED --ex NFT
```

**NOTE:** Before running eval.py make sure to create a `simclr-ecg-eval/transfereval-{NFT}ex` to store the evaluation logs.

# 👨🏻‍⚖️ License

## 🧑🏼‍💻 + 👨🏻‍⚖️ Code License

The original repository was licensed with Apache License, Version 2.0, and therfore as extension this codebase and model weights as Weights & Biases Artifacts are also released under the same license 

## 💽 + 👨🏻‍⚖️ Dataset License

The dataset used for reproducing the results of the paper i.e. [**"PTB-XL, a large publicly available electrocardiography dataset"**](https://www.physionet.org/content/ptb-xl/1.0.1/) is released under the Creative Commons Attribution 4.0 International Public License, whose copy can be obtained [here](https://creativecommons.org/licenses/by/4.0/legalcode). Thus, when downloading the dataset, you agree to the terms mentioned in the afforementioned license.