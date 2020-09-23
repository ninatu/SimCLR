# SimCLR for pre-training VGG19 weights

Fork of [PyTorch SimCLR](https://github.com/sthalles/SimCLR) adapted for pre-training VGG19 weights 
for usage in [Medical Out-of-Distribution Analysis Challenge (MOOD)](http://medicalood.dkfz.de/web/).

See original code and documentation in [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)

See a description of our solution to the MOOD Challenge in 
[https://github.com/ninatu/mood_challenge](https://github.com/ninatu/mood_challenge)

## Usage

1. Install the anomaly detection framework
```
pip install git+https://github.com/ninatu/mood_challenge.git
```
2. Perform "Data Preparation" step. See [git@github.com:ninatu/mood_challenge.git](git@github.com:ninatu/mood_challenge.git)
3. Put correct paths in `config/exp_1_mood.yaml` and run 
```
python run.py configs/exp_1_mood_tmp.yaml
```

