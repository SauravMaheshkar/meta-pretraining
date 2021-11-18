# Meta-Parameterized SimCLR
This folder contains code to run the meta-parameterized SimCLR experiments in the paper.


## Setting up the environment

1. Create and activate a python virtual environment
```python3
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

2. Download the dataset using the provided bash file
```bash
./download_dataset.sh
```

# Pre-training

**No augmentation learning:** To train a SimCLR model with default augmentations run:

```bash
python3 train.py --warmup_epochs 100 --epochs 50 --teacherarch warpexmag --studentarch resnet34 --seed <SEED>
```

The warmup epochs being greater than the number of epochs means the augmentations are not optimized.


**Augmentation learning:** To train a SimCLR model and optimize augmentations, with `N` MetaFT examples, run:

```bash
python3 train.py --warmup_epochs 1 --epochs 50 --teacherarch warpexmag --gpu GPU --seed SEED --ex N
```



# Fine-tuning

To fine-tune a pre-trained model on `NFT` fine-tuning examples (FT dataset has `N` data points), with FT seed RUNSEED and dataset seed (i.e., PT seed) SEED:

```bash
python simclr_eval.py --gpu GPU --checkpoint /PATH/TO/CHECKPOINT --transfer_eval --runseed RUNSEED --seed SEED --ex NFT
```

Note: The partial FT access setting is when `NFT` is more than `N` from the PT.
