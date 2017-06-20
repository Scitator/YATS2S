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
* TF.Estimators
* tensorboard integration
* and finally: best-practices for data input pipelines, let's make it quick!

### Inspired by:
* [google/seq2seq](https://github.com/google/seq2seq)

### Requirements:
Tensorflow 1.2

#### Example usage
* [notebook](https://github.com/Scitator/TF-seq2seq/blob/versions/tf_1.2/seq2seq_example.ipynb)

##### Step-by-step guide:

1. find some parallel corpora
    * for example, let's take en-ru pair from [here](http://www.manythings.org/anki/)
2. prepare it for training (preprocessing and vocabulary extraction)
    * quite simple with [this repo](https://github.com/Scitator/subword-nmt)
    ```bash
       sh prepare_parallel_data.sh --data ./data/tatoeba_en_ru/en_ru.txt --clear_punctuation --lowercase \
           --level bpe --bpe_symbols 10000 --bpe_min_freq 5 \
           --vocab_min_freq 5 --vocab_max_size 10000 \ 
           --merge_sequences --test_ratio 0.1 --clear_tmp
    ```
3. run training process
    * around 100 epochs for this example
    ```bash
       rm -r ./logs_170620_tatoeba_en_ru; python train_parallel_corpora.py \
           --train_corpora_path ./data/tatoeba_en_ru/train.txt --test_corpora_path ./data/tatoeba_en_ru/test.txt \
           --vocab_path ./data/tatoeba_en_ru/vocab.txt \
           --embedding_size 128 --num_units 128 --cell_num 1 \
           --attention bahdanau --residual_connections --residual_dense \
           --training_mode scheduled_sampling_embedding --scheduled_sampling_probability 0.2 \
           --batch_size 64 --queue_capacity 1024 --num_threads 1 \
           --log_dir ./logs_170620_tatoeba_en_ru \
           --train_steps 758200 --eval_steps 842 --min_eval_frequency 7582 \
           --gpu_option 0.8
    ```
4. run tensorboard (really helpful tool: figures, embeddings, graph structure)
    ```bash
       tensorboard --logdir=./logs_170620_tatoeba_en_ru
    ```
5. look at the results
 * [data](https://drive.google.com/file/d/0ByLYAV32riyVa3JxXzZEMXBOWVU/view?usp=sharing)
 * [model](https://drive.google.com/file/d/0ByLYAV32riyVcG1aaEVBUmpnNkE/view?usp=sharing)
 * [notebook](https://github.com/Scitator/TF-seq2seq/blob/versions/tf_1.2/seq2seq_example_tatoeba_inference.ipynb)

#### Contributing

If you find an issue - you know what to do.
