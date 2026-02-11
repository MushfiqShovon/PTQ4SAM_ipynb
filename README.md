# Labmate Quickstart (Notebook)

This repo is set up to run the quantized SAM evaluation from the notebook.
The notebook will auto-download the quantized predictor if it is missing.

## 1) Clone the repo
```
git clone https://github.com/MushfiqShovon/PTQ4SAM_ipynb.git
cd PTQ4SAM/PTQ4SAM
```

## 2) Create the environment
Recommended (lighter setup):
```
conda create -n ptq4sam python=3.8 -y
conda activate ptq4sam
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install "mmcv-full==1.7.1"
pip install -r requirements.txt
```

Build ops + mmdet:
```
cd projects/instance_segment_anything/ops
python setup.py build install
cd ../../..

cd mmdetection
python setup.py build develop
cd ..
```

Optional (exact environment pin):
```
pip install -r requirements/ptq4sam-freeze.txt
```

## 3) Prepare data + checkpoints
- Put COCO val2017 under `data/coco` (annotations + val2017 images)
- Put required detector weights under `ckpt/` (YOLOX-L)

## 4) Run the notebook
Open `quant_infer.ipynb` and run cells top-to-bottom.
The notebook will download the quantized predictor if it is missing:
- Google Drive file id: 103Y5SadTarO4obWzNnWb0jVDnVzrGn3j

## Notes
- The notebook defaults to a 10-image COCO subset for faster checks.
- The quantized predictor is stored at:
  `result/yolox_l_vitb_w6a6/quant_sam_predictor.pth`
