import numpy as np
from chainer.dataset.iterator import Iterator
from chainer.serializers import DictionarySerializer
from nose.tools import raises, assert_equal, assert_true, assert_not_equal

from shoelace.iterator import LtrIterator
from test.utils import get_dataset


def test_iterations():

    # Sample dataset
    dataset = get_dataset()

    # Iterator
    it = LtrIterator(dataset, repeat=False, shuffle=False)
    assert isinstance(it, Iterator)
    assert isinstance(it, LtrIterator)

    # Get all items from the iterator and check their values
    items = list(it)
    assert_equal(len(items), 3)
    assert_equal(len(items[0]), 6)
    assert_equal(len(items[1]), 9)
    assert_equal(len(items[2]), 10)


def test_repeat_true():

    # Sample dataset
    dataset = get_dataset()

    # Iterator
    it = LtrIterator(dataset, repeat=True, shuffle=False)

    counter = 0
    while it.epoch_detail <= 2.0:
        counter += 1
        it.next()

    assert_equal(counter, 7)


@raises(StopIteration)
def test_repeat_false():

    # Sample dataset
    dataset = get_dataset()

    # Iterator
    it = LtrIterator(dataset, repeat=False, shuffle=False)

    # Attempt to iterate beyond the dataset size with repeat set to False
    counter = 0
    while it.epoch_detail <= 2.0:
        counter += 1
        it.next()

    # We should never reach this state, a StopIteration should've been raised
    assert_true(False)


def test_shuffle_true():

    # Sample dataset
    dataset = get_dataset()

    # Seed randomness for repeatability
    np.random.seed(4100)

    # Iterator
    it = LtrIterator(dataset, repeat=True, shuffle=True)

    assert_equal(len(it.next()), 6)
    assert_equal(len(it.next()), 9)
    assert_equal(len(it.next()), 10)

    assert_equal(len(it.next()), 10)
    assert_equal(len(it.next()), 9)
    assert_equal(len(it.next()), 6)

    assert_equal(len(it.next()), 6)
    assert_equal(len(it.next()), 10)
    assert_equal(len(it.next()), 9)


def test_shuffle_false():

    # Sample dataset
    dataset = get_dataset()

    # Seed randomness for repeatability
    np.random.seed(4100)

    # Iterator
    it = LtrIterator(dataset, repeat=True, shuffle=False)

    assert_equal(len(it.next()), 6)
    assert_equal(len(it.next()), 9)
    assert_equal(len(it.next()), 10)

    assert_equal(len(it.next()), 6)
    assert_equal(len(it.next()), 9)
    assert_equal(len(it.next()), 10)

    assert_equal(len(it.next()), 6)
    assert_equal(len(it.next()), 9)
    assert_equal(len(it.next()), 10)


def test_serialize():

    # Sample dataset
    dataset = get_dataset()

    # Iterator
    it = LtrIterator(dataset, repeat=True, shuffle=False)

    # Set up serializer
    serializer = DictionarySerializer()

    # Serialize
    it.serialize(serializer)
    print(serializer.target['epoch'])

    # Perform one epoch iteration
    it.next()
    it.next()
    it.next()

    # Before serializing, our epoch variable should now be different
    assert_not_equal(serializer.target['epoch'], it.epoch)

    # After serializing it should be equal again
    it.serialize(serializer)
    assert_equal(serializer.target['epoch'], it.epoch)
