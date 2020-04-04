# Speech Enhancement Generative Adversarial Network in PyTorch

### Requirements

```bash
sudo apt-get install ffmpeg
```


Note that some versions of the packages have been removed, as there were errors with the versions here stated. (`pillow` requirement has been included by me, @betegon, as pillow 7.0 gives errors, reference: https://github.com/python-pillow/Pillow/issues/4130)

if problems, remove versions of: numba, numpy, matplotlib and pyfftw.

If it keeps throwing errors, just go one package by one, _e.g.,_ `pip install numpy`, `pip install torch==0.4.1.post2` **NOTE:** It is important to first install numpy and then pytorch if going one by one, if not, pytorch will be installed without numpy, and you will have to install numpy and then reinstall torch.

**NOTE:** `fleep` is just for determine audio extension (just for analytics purposes). This way, we can know what type of audio (wav, mp3, flac) are users uploading the most.
```
pillow==6.1
SoundFile==0.10.2
scipy==1.1.0
librosa==0.6.1
h5py==2.8.0
git+https://github.com/santi-pdp/ahoproc_tools.git
numba==0.38.0
torch==0.4.1
matplotlib==2.2.2
numpy==1.14.3
pyfftw==0.10.4
tensorboardX==1.4
torchvision==0.2.1
fleep
```
Ahoprocessing tools (`ahoproc_tools`) is also needed, and the public repo is found [here](git@github.com:santi-pdp/ahoproc_tools.git). If there is problems with ahoproc_tools, just clone the repo: https://github.com/santi-pdp/ahoproc_tools inside this repo root folder and run `python setup.py install` inside `ahoproc_tools` directory. To check it is well installed just run `python` and `import ahoproc_tools`, there should'nt be any error.

### Audio Samples

Latest denoising audio samples with baselines can be found in the [segan+ samples website](http://veu.talp.cat/seganp/). SEGAN is the vanilla SEGAN version (like the one in TensorFlow repo), whereas SEGAN+ is the shallower improved version included as default parameters of this repo.

The voicing/dewhispering audio samples can be found in the [whispersegan samples website](http://veu.talp.cat/whispersegan). Artifacts can now be palliated a bit more with `--interf_pair` fake signals, more data than the one we had available (just 20 mins with 1 speaker per model) and longer training session by iterating more than `100 epoch`.

### Pretrained Models

SEGAN+ generator weights are released and can be downloaded in [this link](http://veu.talp.cat/seganp/release_weights/segan+_generator.ckpt). Make sure you place this file into the `ckpt_segan+/` directory to make it work with the proper `train.opts` config file within that folder. The script `run_segan+_clean.sh` will properly read the ckpt in that directory as it is configured to be used with this referenced file.

So the weights name when downloading them is  `segan+_generator.ckpt`, and and its path should be `ckpt_segan+/segan+_generator.ckpt`. Also, in  that same path, it should be the file `train.opts`, which is located in this repository.



### Introduction to scripts

Two models are ready to train and use to make wav2wav speech enhancement conversions. SEGAN+ is an
improved version of SEGAN [1], denoising utterances with its generator network (G).

![SEGAN+_G](assets/segan+.png)

To train this model, the following command should be ran:

```
python train.py --save_path ckpt_segan+ --batch_size 300 \
		--clean_trainset data/clean_trainset \
		--noisy_trainset data/noisy_trainset \
		--cache_dir data/cache
```

Read `run_segan+_train.sh` for more guidance. This will use the default parameters to structure both G and D, but they can be tunned with many options. For example, one can play with `--d_pretrained_ckpt` and/or `--g_pretrained_ckpt` to specify a departure pre-train checkpoint to fine-tune some characteristics of our enhancement system, like language, as in [2].

Cleaning files is done by specifying the generator weights checkpoint, its config file from training and appropriate paths for input and output files (Use `soundfile` wav writer backend (recommended) specifying the `--soundfile` flag):

```
python clean.py --g_pretrained_ckpt ckpt_segan+/<weights_ckpt_for_G> \
		--cfg_file ckpt_segan+/train.opts --synthesis_path enhanced_results \
		--test_files data/noisy_testset --soundfile
```

Read `run_segan+_clean.sh` for more guidance.

There is a WSEGAN, which stands for the dewhispering SEGAN [3]. This system is activated (rather than vanilla SEGAN) by specifying the `--wsegan` flag. Additionally, the `--misalign_pair` flag will add another fake pair to the adversarial loss indicating that content changes between input and output of G is bad, something that improved our results for [3].

### References:

1. [SEGAN: Speech Enhancement Generative Adversarial Network (Pascual et al. 2017)](https://arxiv.org/abs/1703.09452)
2. [Language and Noise Transfer in Speech Enhancement GAN (Pascual et al. 2018)](https://arxiv.org/abs/1712.06340)
3. [Whispered-to-voiced Alaryngeal Speech Conversion with GANs (Pascual et al. 2018)](https://arxiv.org/abs/1808.10687)

### Cite

```
@article{pascual2017segan,
  title={SEGAN: Speech Enhancement Generative Adversarial Network},
  author={Pascual, Santiago and Bonafonte, Antonio and Serr{\`a}, Joan},
  journal={arXiv preprint arXiv:1703.09452},
  year={2017}
}
```

### Notes

* Multi-GPU is not supported yet in this framework.
* Virtual Batch Norm is not included as in the very first SEGAN code, as similar results to those of original paper can be obtained with regular BatchNorm in D (ONLY D).
* If using this code, parts of it, or developments from it, please cite the above reference.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.
