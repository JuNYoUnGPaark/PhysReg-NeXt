# Learning Physically Consistent Representations for Human Activity Recognition via Modern Temporal Convolutional Network

<p align="center"><img src='./overall_.png'></p>

This repository implements the methodology proposed in the paper "Learning Physically Consistent Representations for Human Activity
Recognition via Modern Temporal Convolutional Network".


## Paper Overview
**Abstract**: 

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
- `model.py` - Implementation of the proposed ~.
The implementation uses PyTorch, Numpy, pandas, scikit-learn, matplotlib, seaborn, and fvcore (for FLOPs analysis).

## Citing this Repository

If you use this code in your research, please cite:

```
@article{Learning Physically Consistent Representations for Human Activity
Recognition via Modern Temporal Convolutional Network,
  title = {Learning Physically Consistent Representations for Human Activity
Recognition via Modern Temporal Convolutional Network},
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
