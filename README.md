# Dolus

This is a neural network for evaluating password strength.

Features:
- Based on GRU architecture
- **98,8%** validation accuracy
- Uses a combination of length, entropy, and features detected by the neural network
- Small enough to be run in constrained environments
- Pretrained models are provided for Tensorflow and Onxx

## Running

The pre-trained model file can be found from `/model/`. A simple script `run.py` can be used to test the model.

```bash
$ python3 run.py ../model/gru_dolus.keras mypassword
The string 'mypassword' is classified as BAD.
$ python3 run.py ../model/gru_dolus.keras hungryhungryhipposlovepelicansandmushrooms7
The string 'hungryhungryhipposlovepelicansandmushrooms7' is classified as GOOD.
```

## Training process

You need to do this only if you want to develop the model further.

### Generating the data sets

The datasets are too large for github. If you need them you have to generate them.

The **BAD** dataset is `rockyou.txt`. Because it seems to by default contain invalid utf-8 characters it has to be cleaned up. Also shuffle the lines using `shuf`.

```bash
$ iconv -f utf-8 -t utf-8 -c rockyou.txt -o bad.txt
$ shuf --random-source=/dev/urandom bad.txt > bad_shuffled.txt
```

The **GOOD** dataset is entirely synthetic and created with `generate.py`. Again shuffle the lines using `shuf`.

```bash
$ python generate.py good.txt 15000000
$ shuf --random-source=/dev/urandom good.txt > good_shuffled.txt
```

After that preprocess the datasets and save them in tfrecord file format.

```bash
$ python3 prep_datasets.py good_shuffled.txt bad_shuffled.txt train_ds.tfrecord val_ds.tfrecord
```

### Training

After the data is in tfrecord format you can train the neural network.

```bash
$ python3 train.py train_ds.tfrecord val_ds.tfrecord
```

The script will output the following files:
- **dolus.png** summary for the model
- **accuracy_plot.png** validation accuracty plot
- **loss_plot.png** loss plot
- **dolus.keras** the trained model
- **snapshot.keras** snapshot of the weighs in the best performing epoch


