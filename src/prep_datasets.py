import tools
import tensorflow as tf
import argparse

max_length = 64

parser = argparse.ArgumentParser(prog='prep_datasets.py', description='Prepares the datasets for Tensorflow training')
parser.add_argument('ingood', help="Input file containing good samples")
parser.add_argument('inbad', help="Input file containing bad samples")
parser.add_argument('trainds', help="Output file containing training dataset")
parser.add_argument('valds', help="Output file containing validation dataset")
parser.parse_args()
args = parser.parse_args()

# Create datasets for good and bad passwords
good_dataset = tools.create_dataset(args.ingood, max_length=max_length)
bad_dataset = tools.create_dataset(args.inbad, max_length=max_length)

# Combine and label the datasets
good_dataset = tools.add_labels(good_dataset, 1)  # Label 1 for good passwords
bad_dataset = tools.add_labels(bad_dataset, 0)  # Label 0 for bad passwords

# Combine and interleave the datasets
interleaved_dataset = tf.data.Dataset.sample_from_datasets(
    [good_dataset, bad_dataset],
    weights=[0.5, 0.5],  # Equal probability for both datasets
    stop_on_empty_dataset=True  # Continue even if one dataset is exhausted
)

# Filter dataset correctly while maintaining structure & split it
split_ds = interleaved_dataset.map(lambda x, y: tools.assign_split(x, y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = split_ds.filter(lambda x, y, split_label: tf.equal(split_label, "train")).map(lambda x, y, _: (x, y))
val_ds = split_ds.filter(lambda x, y, split_label: tf.equal(split_label, "val")).map(lambda x, y, _: (x, y))

tools.save_to_tfrecord(train_ds, args.trainds)
tools.save_to_tfrecord(val_ds, args.valds)
