

## 📦 Get started

### Environment Preparing
```
conda create -n evi-cem python=3.10
conda activate evi-cem
# please modify according to the CUDA version in your server
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Dataset Preparing
1. Run the following code to download `fitzpatrick17k.csv`
```
wget -P data/meta_data https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv
```
2. Run the following code to download the SkinCon Fitzpatrick17k annotation
```
wget -O data/meta_data/skincon.csv https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv
```
3. Run `python data/raw_data_download.py` to download the fitzpatrick17k images. If any image links in the `fitzpatrick17k.csv` become invalid, the raw images can be downloaded [here](https://drive.google.com/file/d/1Eb7MGGr1Dj0z2xgEuMuCoblECuPDCrhD/view?usp=share_link)
4. Run `python data/generate_clip_concepts.py` to generate soft concept labels with [MONET](https://github.com/suinleelab/MONET)

### evi-CEM training

**Under complete concept supervision**:
```
python train.py --config configs/default.yaml
```
**Label-efficient training**:
```
python train.py --config configs/label_efficient.yaml
```
**Label-efficient training with concept rectification**:
```
python learn_cavs.py --config configs/learn_cavs.yaml
python train.py --config configs/train_rectified.yaml
```

