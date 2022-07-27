# MDKT

The implementaion of paper [Multi-directional Knowledge Transfer for Few-Shot Learning](http://staff.ustc.edu.cn/~hexn/papers/mm22-knowledge-few-shot.pdf), which has been accepted by  ACM Multimedia 2022.
## Requirements

- Python3
- PyTorch1.8, torchvision
## Running the code
Our extracted features from **ImNet** dataset are available at [onedrive](https://1drv.ms/f/s!At7V-cd4rg6msQ1YasLNLolhydmj). _Download the extracted features and place them into an empty directory Features/_.

The benchmark  of **ImNet** dataset with 5 different settings for the number of novel category examples K = {1,2,3,4,5}. The main entry point for running the low shot benchmark is`low_shot.py`, which will run a single experiment for a single value of _K_. (Refer to [Low-shot Visual Recognition by Shrinking and Hallucinating Features](https://github.com/facebookresearch/low-shot-shrink-hallucinate))

For example, running the first experiment with _K_=1 will look like:
```shell
python ./low_shot.py --nshot 1 \
  --lr 0.001 --wd 0.001 \
  --batchsize 1000
```

## Performance
| Method | K=1 | K=2 | K=3 | K=4 | K=5 |
| --- | --- | --- | --- | --- | --- |
| baseline | 36.1 | 47.9 | 54.0 | 58.1 | 60.8 |
| MDKT | 44.4 | 53.3 | 58.1 | 61.7 | 63.8 |

*We use a visual classifier composed of a full connection layer  as the baseline model.
