# AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer
Pengjun Fang, Yingqing He, Yazhou Xing, Qifeng Chen, Ser-Nam Lim, Harry Yang

The Hong Kong University of Science and Technology, University of Central Florida

ICLR 2026

## Environment Setup
```bash
conda create -n acfoley python==3.10 -y
conda activate acfoley
git clone https://github.com/ff2416/AC-Foley.git
cd AC-Foley
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install .
```
## Download Weights
```bash
bash scripts/download.sh
```
## Inference
```bash
bash scripts/demo.sh
```
