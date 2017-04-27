# ANA-Dialogue-Generation

## Training Instructions
### Seq2Seq with attention mechanism
```sh
> mkdir params
> python dialoge.py --attention True
```
### Seq2Seq without attention mechanism
```sh
> mkdir params
> python dialoge.py --attention False
```
## Testing Instructions
### Evaluating on the test dataset (Here, the training was done with attention mechanism)
```sh
> python dialoge.py --attention True --decode_file --train_dir .\params\
```
### Interactive evaluating (Here, the training was done without attention mechanism)
```sh
> python dialoge.py --attention False --decode_shell --train_dir .\params\
```

please see the flags of the `dilogue.py` file for more hyper parameters.
## Downloading the Dataset
Download the preprocessed version of the dataset from this link:
https://drive.google.com/file/d/0B_18bOXmh2WJNVZTOWw2SDRPYTg/view?usp=sharing

After downloading and unziping, the `data` directory will contain the following files:

`train.q`: questions (given utternaces) for training dataset

`train.r`: true responses for training dataset

`dev.q`: questions (given utternaces) for development dataset

`dev.r`: responses for developement dataset

`test.q`: questions (given utternaces) for test dataset

`test.r`: responses for test dataset

`questions.q` designed questions (utterances) for manual evaluation

## Requirements
``Tensorflow version r1.0``, ``NLTK 3.0`` and ``Python 2.7.x``
## License
`data_utils.py`, `seq2seq.py`, `seq2seq_model.py` and `dialogue.py` are the modified versions of the Tensorflow's legacy implementation for the seqeunce to sequence model. They are being published under the Apache License, Version 2.0. Changes in each file are specified as comments.

`bleu.py` is released under MIT license.
