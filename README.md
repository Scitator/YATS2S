# YATS2S: Yet Another Tensorflow Seq2seq

So, here you are stranger. Finally, you found it!

### Overview
After some time of looking around for user-friendly and configurable seq2seq TF implementation, I decided to make my own one.

So, here it is:
* Pure TF
* any cell you want - just say it name
* multi-layer
* bidirectional
* attention
* residual connections, residual dense
* and other seq2seq cells tricks available!
* vocabulary trick: joint or different for source and target?
* scheduled_sampling
* in-graph beam search
* and finally: best-practices for data input pipelines, let's make it quick!

### Inspired by:
* [google/seq2seq](https://github.com/google/seq2seq)

### Requirements:
Tensorflow 1.2

#### Example usage
* [notebook](https://github.com/Scitator/TF-seq2seq/blob/versions/tf_1.2/seq2seq_example.ipynb)

#### Contributing

If you find an issue - you know what to do.
