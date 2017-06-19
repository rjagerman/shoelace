from io import StringIO, BytesIO

import numpy as np
from nose.tools import raises, assert_equal, assert_in, assert_not_equal, \
    assert_true

from shoelace.dataset import LtrDataset
from test.utils import get_dataset


def test_save_txt_and_load_txt():

    # Get sample data set
    dataset = get_dataset()

    # Get in-memory string handle
    with StringIO() as handle:

        # Save text to handle
        dataset.save_txt(handle)
        handle.seek(0)

        # Load text from handle
        dataset2 = LtrDataset.load_txt(handle)

        # Assert that everything loaded correctly
        assert_true(np.array_equal(dataset.feature_vectors,
                                   dataset2.feature_vectors))
        assert_true(np.array_equal(dataset.relevance_scores,
                                   dataset2.relevance_scores))
        assert_true(np.array_equal(dataset.query_pointer,
                                   dataset2.query_pointer))
        assert_true(np.array_equal(dataset.query_ids, dataset2.query_ids))


def test_save_and_load():

    # Get sample data set
    dataset = get_dataset()

    # Get in-memory binary handle
    with BytesIO() as handle:

        # Save binary to handle
        dataset.save(handle)
        handle.seek(0)

        # Load binary from handle
        dataset2 = LtrDataset.load(handle)

        # Assert that everything loaded correctly
        assert_true(np.array_equal(dataset.feature_vectors,
                                   dataset2.feature_vectors))
        assert_true(np.array_equal(dataset.relevance_scores,
                                   dataset2.relevance_scores))
        assert_true(np.array_equal(dataset.query_pointer,
                                   dataset2.query_pointer))
        assert_true(np.array_equal(dataset.query_ids, dataset2.query_ids))


def test_get_sample():

    # Get sample data set
    dataset = get_dataset()

    # Assert that splitting per item works
    assert_equal(dataset[0].feature_vectors.shape, (6, 45))
    assert_equal(dataset[0].relevance_scores.shape, (6, 1))
    assert_equal(dataset[0].nr_queries, 1)
    assert_equal(dataset[0].query_ids, ['1'])

    assert_equal(dataset[1].feature_vectors.shape, (9, 45))
    assert_equal(dataset[1].relevance_scores.shape, (9, 1))
    assert_equal(dataset[1].nr_queries, 1)
    assert_equal(dataset[1].query_ids, ['16'])

    assert_equal(dataset[2].feature_vectors.shape, (10, 45))
    assert_equal(dataset[2].relevance_scores.shape, (10, 1))
    assert_equal(dataset[2].nr_queries, 1)
    assert_equal(dataset[2].query_ids, ['63'])


@raises(IndexError)
def test_get_sample_out_of_range():

    # Get sample data set
    dataset = get_dataset()

    # Raise an exception by trying to get an element out of range
    item = dataset[3]

    # This state should never be reached
    assert False


@raises(IndexError)
def test_get_sample_out_of_range():

    # Get sample data set
    dataset = get_dataset()

    # Raise an exception by trying to get an element out of range
    item = dataset[-1]

    # This state should never be reached
    assert_true(False)


def test_slicing():

    # Get sample data set
    dataset = get_dataset()

    # Grab a slice
    dataset_slice = dataset[0:2]

    # Assert that the slice indexed the correct elements
    assert_equal(dataset_slice[0].feature_vectors.shape, (6, 45))
    assert_equal(dataset_slice[0].relevance_scores.shape, (6, 1))
    assert_equal(dataset_slice[0].nr_queries, 1)
    assert_equal(dataset_slice[0].query_ids, ['1'])

    assert_equal(dataset_slice[1].feature_vectors.shape, (9, 45))
    assert_equal(dataset_slice[1].relevance_scores.shape, (9, 1))
    assert_equal(dataset_slice[1].nr_queries, 1)
    assert_equal(dataset_slice[1].query_ids, ['16'])

    assert_equal(len(dataset_slice), 2)

    # Grab another slice
    dataset_slice = dataset[1:3]

    # Assert that the slice indexed the correct elements
    assert_equal(dataset_slice[0].feature_vectors.shape, (9, 45))
    assert_equal(dataset_slice[0].relevance_scores.shape, (9, 1))
    assert_equal(dataset_slice[0].nr_queries, 1)
    assert_equal(dataset_slice[0].query_ids, ['16'])

    assert_equal(dataset_slice[1].feature_vectors.shape, (10, 45))
    assert_equal(dataset_slice[1].relevance_scores.shape, (10, 1))
    assert_equal(dataset_slice[1].nr_queries, 1)
    assert_equal(dataset_slice[1].query_ids, ['63'])

    assert_equal(len(dataset_slice), 2)


def test_len():

    # Get sample data set
    dataset = get_dataset()

    # Assert the len operator works
    assert_equal(len(dataset), 3)
    assert_equal(len(dataset), dataset.nr_queries)
    assert_equal(len(dataset[0]), 1)


def test_normalize_true():

    # Get sample data set
    dataset = get_dataset(normalize=True)

    # Check every feature has a correct maximum and minimum
    for i in range(len(dataset)):
        per_feature_max = np.max(dataset[i].feature_vectors, axis=0)

        # Maximum can be zero, in cases where the data has no range to normalize
        # by (i.e. when all features are always zero)
        assert_in(np.min(per_feature_max), (0.0, 1.0))
        per_feature_min = np.min(dataset[i].feature_vectors, axis=0)
        assert_equal(np.max(per_feature_min), 0.0)


def test_normalize_false():

    # Get sample data set
    dataset = get_dataset(normalize=False)

    # Check every feature has a correct maximum and minimum
    for i in range(len(dataset)):
        per_feature_max = np.max(dataset[i].feature_vectors, axis=0)

        assert_not_equal(np.min(per_feature_max), 1.0)
        per_feature_min = np.min(dataset[i].feature_vectors, axis=0)
        print(per_feature_min)
        assert_not_equal(np.max(per_feature_min), 0.0)
