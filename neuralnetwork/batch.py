import numpy as np

from math import ceil


class Batch(object):
    def __call__(cls, inputs, outputs, batch_size=10):
        assert len(inputs) == len(outputs)
        assert batch_size > 0

        intervals = range(int(ceil(1.0 * len(inputs) / batch_size)))
        np.random.shuffle(intervals)

        for start in intervals:
            end = min(start + batch_size, len(inputs))
            yield (inputs[start:end], outputs[start:end])
