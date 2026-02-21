# Physics-Guided Auxiliary Learning for Robust HAR via Multi-Scale NeXt-TCN

<p align="center"><img src='./overall_.png'></p>

This repository implements the methodology proposed in the paper "Physics-Guided Auxiliary Learning for Robust HAR via Multi-Scale NeXt-TCN".


## Paper Overview
**Abstract**: Deep learning-based Human Activity Recognition (HAR)
with wearable IMUs often struggles with generalization due to the
reliance on dataset-specific statistics rather than consistent motion
dynamics. We propose a Physics-Guided Auxiliary Learning
framework built on a novel Multi-Scale NeXt-TCN backbone. NeXt-
TCN combines multi-scale depthwise-separable temporal convolutions
and large-kernel context modules to capture both shortterm
transitions and long-range dependencies with high efficiency.
During training only, a lightweight physics-guided auxiliary branch
enforces kinematic constraints derived from gravity alignment,
complementary filtering, and rigid-body dynamics to encourage
physically consistent feature representations; this improves robustness
without increasing inference cost. On four public datasets (UCI-HAR, WISDM, PAMAP2, MHEALTH), our method
achieves highly competitive performance (e.g., Macro F1 = 99.27% on PAMAP2) and establishes a state-of-the-art tradeoff
between accuracy and computational efficiency, remaining compact ( 0.07M) and fast ( 3 ms on a CPU workstation and
28 ms on a Raspberry Pi 4). These results demonstrate that embedding physical inductive biases via auxiliary supervision
substantially improves both generalization and representation quality for wearable-sensor HAR.

## Dataset
- **UCI-HAR** dataset is available at _https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones_
- **PAMAP2** dataset is available at _https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring_
- **MHEALTH** dataset is available at _https://archive.ics.uci.edu/dataset/319/mhealth+dataset_
- **WISDM** dataset is available at _https://www.cis.fordham.edu/wisdm/dataset.php_

## Requirements
```
torch==2.6.0+cu126
numpy==2.3.4
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.9.2
seaborn==0.13.2
fvcore==0.1.5.post20221221
```
To install all required packages:
```
pip install -r requirements.txt
```

## Codebase Overview
- `model.py` - Implementation of the proposed **Physics-Guided Auxiliary Learning framework (PhysReg-NeXt)**.
The implementation uses PyTorch, Numpy, pandas, scikit-learn, matplotlib, seaborn, and fvcore (for FLOPs analysis).

## Citing this Repository

If you use this code in your research, please cite:

```
@article{Physics-Guided Auxiliary Learning for Robust HAR via Multi-Scale NeXt-TCN,
  title = {Physics-Guided Auxiliary Learning for Robust HAR via Multi-Scale NeXt-TCN},
  author={JunYoung Park and Myung-Kyu Yi}
  journal={},
  volume={},
  Issue={},
  pages={},
  year={}
  publisher={}
}
```

## Contact

For questions or issues, please contact:
- JunYoung Park : park91802@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
