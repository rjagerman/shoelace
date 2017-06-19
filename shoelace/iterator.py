import numpy as np
from chainer.dataset import iterator
from chainer.serializer import Serializer


class LtrIterator(iterator.Iterator):
    """Dataset iterator that serially reads learning-to-rank examples.

    This is an implementation of :class:`~chainer.dataset.Iterator` that visits
    each query of a :class:`shoelace.dataset.dataset.LtrDataset` object and
    generates a variable-sized minibatch of the query-document instances for
    that query.

    This means that each minibatch contains all the documents for a particular
    query, which can be of varying sizes.

    Args:
        dataset: Dataset ot iterate.
        repeat: Whether to repeat iterations over the data set (default: False)

    """

    def __init__(self, dataset, repeat = False, shuffle= True):
        self.feature_vectors = dataset.feature_vectors
        self.query_pointer = dataset.query_pointer
        self.relevance_scores = dataset.relevance_scores.astype(np.float32)
        self._shuffle = shuffle
        self._nr_of_queries = len(dataset)
        self._query_index = np.arange(0, self._nr_of_queries)
        self._repeat = repeat
        self.reset()

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration
        self._previous_epoch_detail = self.epoch_detail

        start = self.query_pointer[self._query_index[self._current_index]]
        end = self.query_pointer[self._query_index[self._current_index] + 1]

        self.batch_size = end - start
        self._current_index += 1

        if self._current_index >= self._nr_of_queries:
            self._current_index = 0
            self.epoch += 1
            if self._shuffle:
                self._shuffle_indices()

        return [(self.feature_vectors[i, :], self.relevance_scores[i]) for
                i in range(start, end)]

    def _shuffle_indices(self):
        """
        Shuffles the indices so the next iteration iterates the data in a
        different order 
        """
        self._query_index = np.random.permutation(self._nr_of_queries)

    @property
    def epoch_detail(self):
        return self.epoch + self._current_index / self._nr_of_queries

    def serialize(self, serializer):
        self.batch_size = serializer('batch_size', self.batch_size)
        self.epoch = serializer('epoch', self.epoch)
        self.previous_epoch_detail = serializer('previous_epoch_detail',
                                                self.previous_epoch_detail)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        self._current_index = serializer('_current_index', self._current_index)

    def reset(self):
        self.batch_size = 0
        self.epoch = 0
        self.previous_epoch_detail = None
        self.is_new_epoch = False
        self._current_index = 0
