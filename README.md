<div align="center">

# AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer

**Pengjun Fang, Yingqing He, Yazhou Xing, Qifeng Chen, Ser-Nam Lim, Harry Yang**

**The Hong Kong University of Science and Technology, University of Central Florida**

**ICLR 2026**

<p>
  <a href="https://ff2416.github.io/AC-Foley-Page/"><strong>Webpage</strong></a> |
  <a href="https://openreview.net/forum?id=URPXhnWdBF"><strong>Paper</strong></a>
</p>

</div>

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
## Acknowledgement
We would like to thank the authors of these repos for their contribution.
- [MMAudio](https://github.com/hkchengrex/MMAudio?tab=readme-ov-file)
- [Synchformer](https://github.com/v-iashin/Synchformer)
## Citation

```bibtex
@inproceedings{fang2026acfoley,
  title={AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer},
  author={Fang, Pengjun and He, Yingqing and Xing, Yazhou and Chen, Qifeng and Lim, Ser-Nam and Yang, Harry},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
