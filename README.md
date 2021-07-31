# Combined Ranking & Regression for Monocular Depth Estimation

**Disclaimer:** This code is based on the project by Alhashim et al. (2018) which can be found here [DenseDepth](https://github.com/ialhashim/DenseDepth)

## Requirements
- This code is tested using Python 3.7.10, Tensorflow 2.4.1, CUDA 10.1 on a machine with a NVIDIA GTX 1080, 16 GB RAM running on Ubuntu 20.04
- Other packages needed can be found in the [requirements](https://git.uni-paderborn.de/dstecher/combined-ranking-and-regression-for-monocular-depth-esitmation/-/blob/master/requirements.txt) or [conda env yaml](https://git.uni-paderborn.de/dstecher/combined-ranking-and-regression-for-monocular-depth-esitmation/-/blob/master/combdepth.yml) file.
- For compatibility reasons, please set the batch size to 4 while training to ensure correct computations

## Data
- Download the following zip files and place them in the main directory of this repository (for different locations, please adjust data paths where indicated in the code)
- [Dataset NYU Depth V2 (50K)](https://tinyurl.com/nyu-data-zip)
- [Test Set NYU Depth V2](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) (currently corrupted)

## Training
- Run `python train.py --name <model name> --lr 0.0001 --maxlr 0.00357 --bs 4 --lambda_val 0.5` (assuming a training with a weighting of λ)

## Evaluation
- Run `python evaluate.py --model <model name>.h5`

## Test Example Creation
- Run `python test.py --model <model name>.h5`

## Results
![Depths Results](https://git.uni-paderborn.de/dstecher/combined-ranking-and-regression-for-monocular-depth-esitmation/-/blob/master/results_depths_comparison.png?raw=true)
