# Graph-of-Gradient
This is the official code for "Fairness without Demographics on Electronic Health Records" published in AAAI 2024 SSS Clinical Foundation Models [[paper](https://openreview.net/forum?id=5NJp8WZ0Dn)].

## Datasets
MIMIC III: https://physionet.org/content/mimiciii/1.4/  
MIMIC IV: https://physionet.org/content/mimiciv/0.4/  
Diagnoses and Procedures.

## Getting Started
Run data preprocessing following [CHE](https://github.com/yingtaoluo/Causal-Healthcare-Emebedding/). Then, run [Normal/train.py](https://github.com/yingtaoluo/Graph-of-Gradient/blob/main/Normal/train.py) for testing *baselines* and run [train.py](https://github.com/yingtaoluo/Graph-of-Gradient/blob/main/train.py) for testing *GoG*.

## Citation 
If your paper benefits from this repo, please consider citing us:

```
Luo, Yingtao, et al. "Fairness without Demographics on Electronic Health Records." AAAI 2024 Spring Symposium on Clinical Foundation Models. 2024.
```

```bibtex
@inproceedings{luo2024fairness,
  title={Fairness without Demographics on Electronic Health Records},
  author={Luo, Yingtao and Li, Zhixun and Liu, Qiang and Zhu, Jun},
  booktitle={AAAI 2024 Spring Symposium on Clinical Foundation Models},
  year={2024}
}
```

## Test for yourself
Here, we show an example result on only a single demogrpahic: race. This is a simpler setting (only a few subpopulation groups) but a standard setting adopted by most previous works. Let us run on LSTM as the baseline for this example.

For the same parameters (tuned after grid search), specifically, learning rate 3e-3, the results are:  
(For overall acc/ndcg: [@10, @20]. For ethnicity_acc/ndcg: Row ['WHITE', 'BLACK', 'HISPANIC'], Column [@10, @20].)

**Normal (Basline):**

Final test_acc:[0.17643908 0.26110085], test_ndcg:[0.26656379 0.27345939]

Normal test_ethnicity_acc:[[0.17065003 0.25612097]
 [0.20209396 0.28453418]
 [0.17187903 0.25401088]]

**Fair (Baseline + GoG algorithm):**

Final test_acc:[0.23032734 0.32407207], test_ndcg:[0.34178205 0.34083014],

Fair test_ethnicity_acc:[[0.22541894 0.31805558]
 [0.25179777 0.34817921]
 [0.22711514 0.32526572]]

**Analysis**

GoG does not only improve algorithmic fairness, but also helps with convergence, since it is a robust model that alleviates noises.
