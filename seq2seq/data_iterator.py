import threading

import tensorflow as tf


def load_generic_text(fname, process_fn):
    """Generator that yields text raw from the file and encode with vocab"""
    with open(fname, 'r') as fin:
        for line in fin:
            data = process_fn(line)
            yield data


class IteratorQueue(object):
    def __init__(self,
                 iterator,
                 coord,
                 placeholders,
                 queue_size=1024):
        self.iterator = iterator
        self.coord = coord
        self.threads = []

        self.placeholders = placeholders

        self.queue = tf.FIFOQueue(
            queue_size,
            list(map(lambda x: x.dtype, self.placeholders)),
            shapes=list(map(lambda x: x.shape.as_list(), self.placeholders)))

        self.enqueue = self.queue.enqueue(self.placeholders)

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        stop = False
        while not stop:
            for data in self.iterator:
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                sess.run(
                    self.enqueue,
                    feed_dict=dict(zip(self.placeholders, data)))

    def stop_threads(self):
        for t in self.threads:
            t.stop()

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
