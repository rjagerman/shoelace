import re
import numpy as np
import pickle
from collections import defaultdict
from chainer.dataset.dataset_mixin import DatasetMixin


class LtrDataset(DatasetMixin):

    """
    Implementation of Learning to Rank data set

    Supports efficient slicing on query-level data. Note that single samples are
    collections of query-document pairs represented as a tuple of matrix of
    feature vectors and a vector of relevance scores

    """

    def __init__(self, feature_vectors, relevance_scores, query_pointer,
                 query_ids, nr_of_queries):
        self.feature_vectors = feature_vectors
        self.relevance_scores = relevance_scores
        self.query_pointer = query_pointer
        self.query_ids = query_ids
        self.nr_queries = nr_of_queries

    def __len__(self):
        """
        Returns the number of queries.
        """
        return self.nr_queries

    def get_example(self, i):
        """
        Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        if i < 0 or i >= self.nr_queries:
            raise IndexError

        start = self.query_pointer[i]
        end = self.query_pointer[i+1]

        return LtrDataset(self.feature_vectors[start:end, :],
                          self.relevance_scores[start:end], np.zeros(1),
                          [self.query_ids[i]], 1)

    def normalize(self):
        for i in range(self.nr_queries):
            start = self.query_pointer[i]
            end = self.query_pointer[i+1]

            self.feature_vectors[start:end, :] -= np.min(self.feature_vectors[start:end, :], axis=0)
            maximum = np.max(self.feature_vectors[start:end, :], axis=0)
            maximum[maximum == 0.0] = 1.0
            self.feature_vectors[start:end, :] /= maximum

    @classmethod
    def load_txt(cls, file_handle, normalize=False):
        """
        Loads a learning to rank dataset from a text file source
        
        :param filepaths: A single file path as a string or a list of file paths 
        :return: A `class:dataset.dataset.LtrDataset` object
        """

        # Iterate over lines in the file
        data_set = defaultdict(list)
        for line in file_handle:

            # Extract the data point information
            data_point = LtrDataPoint(line)
            data_set[data_point.qid].append(data_point)

        # Convert feature vectors, relevance scores and query pointer to correct
        # form
        query_ids = list(data_set.keys())
        query_pointer = np.array([len(data_set[query]) for query in data_set])
        query_pointer = np.cumsum(query_pointer)
        query_pointer = np.hstack([np.array([0]), query_pointer])
        nr_of_queries = len(data_set)
        feature_vectors = np.vstack([data_point.feature_vector
                                     for query in data_set
                                     for data_point in data_set[query]])
        relevance_scores = np.vstack([data_point.relevance
                                      for query in data_set
                                      for data_point in data_set[query]])

        # Free memory
        del data_set

        # Generate object to return
        result = LtrDataset(feature_vectors, relevance_scores, query_pointer,
                            query_ids, nr_of_queries)

        # If normalization is necessary, do so
        if normalize:
            result.normalize()

        # Cast to float32 (after normalization) which is typical format in
        # chainer
        result.feature_vectors = result.feature_vectors.astype(dtype=np.float32)

        # Return result
        return result

    def save_txt(self, file_handle):
        """
        Saves the data set in txt format to given file
        
        :param file_handle: The file to save to 
        """
        for i in range(self.nr_queries):
            start = self.query_pointer[i]
            end = self.query_pointer[i + 1]
            for j in range(start, end):
                features = " ".join('{i}:{v}'.format(i=i,
                                                     v=self.feature_vectors[j, i])
                                    for i in range(len(self.feature_vectors[j])))
                out = '{r} qid:{qid} {features}\n'.format(r=self.relevance_scores[j,0],
                                                      qid=self.query_ids[i],
                                                      features=features)
                file_handle.write(out)

    def save(self, file_handle):
        """
        Saves the data set in binary format to given file
        
        :param file_handle: The file to save to
        """
        pickle.dump(self, file_handle)

    @classmethod
    def load(cls, file_handle):
        """
        Loads the data set in binary format from given file
        
        :param file_handle: The file to load from 
        :return: A `class:dataset.dataset.LtrDataset` object
        """
        return pickle.load(file_handle)


class LtrDataPoint:
    """
    A single learning to rank data point, contains a query identifier, a
    relevance label and a feature vector
    """

    qid_regex = re.compile(".*qid:([0-9]+).*")
    relevance_regex = re.compile("^[0-9]+")
    feature_regex = re.compile("([0-9]+):([^ ]+)")

    def __init__(self, line):

        # Remove comment
        comment_start = line.find("#")
        if comment_start >= 0:
            line = line[:comment_start]

        # Extract qid
        self.qid = re.search(LtrDataPoint.qid_regex, line).group(1)
        self.relevance = re.search(LtrDataPoint.relevance_regex, line).group(0)
        features = re.findall(LtrDataPoint.feature_regex, line)
        minimum = min(int(index) for index, _ in features)
        maximum = max(int(index) for index, _ in features)
        self.feature_vector = np.zeros(1 + maximum - minimum)
        for index, value in features:
            self.feature_vector[int(index) - minimum] = float(value)
