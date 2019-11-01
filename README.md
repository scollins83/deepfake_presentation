# Few-shot face translation ![Source face: Mona Lisa](https://github.com/shaoanlu/fewshot-face-translation-GAN/raw/master/images/translation_results/MonaLisa_translation.gif)


## Try in Google Colab 

 - `master` branch (Jun. 2019)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaoanlu/fewshot-face-translation-GAN/blob/master/colab_demo.ipynb)

# Otherwise...... 


## Step 1: Download Models
#### download models from links below and move them into a directory call "weights" in the root directory of this repo

https://drive.google.com/uc?id=1DUMmZGTGKMyEYSKy-w34IDHawVF24rIs

https://drive.google.com/uc?id=1xl8cg7xaRnMsyiODcXguJ83d5hwodckB

## Step 2: Install Keras

```py
pip install keras==2.2.4
conda install -c menpo opencv
```

## Step 3: Download Images and add them to 'run.py'

```py
fn_src = "me2.png"
fns_tar = ["download.jpg"]
```


## Step 4: Run the script:
```
python run.py
```






## Requirements
  - Python 3.6
  - Keras 2.2.4
  - TensorFlow 1.12.0 or 1.13.1

## References
1. [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)
2. [Few-Shot Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1905.01723)
3. [DEEP LEARNING FOR FASHION AND FORENSICS](https://drum.lib.umd.edu/handle/1903/21337)

