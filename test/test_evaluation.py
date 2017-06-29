import numpy as np
from nose.tools import raises, assert_equal

from shoelace.evaluation import ndcg


def test_ndcg():

    # Set up data
    prediction = np.array([0.1, 0.9, 0.2, 3.0, 0.15])
    ground_truth = np.array([3.0, 3.0, 2.0, 1.0, 1.0])

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 0.73213389587665278)


def test_ndcg_2():

    # Set up data
    prediction = np.array([0.1, 0.9, 0.2, 0.15, 3.0])
    ground_truth = np.array([3.0, 3.0, 2.0, 1.0, 1.0])

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 0.73213389587665278)


def test_ndcg_3():

    # Set up data
    prediction = np.array([0.1, 0.9, 0.2, 0.15, 3.0])
    ground_truth = np.array([3.0, 3.0, 2.0, 1.0, 2.0])

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 0.8259562683091511)


def test_ndcg_perfect():

    # Set up data
    prediction = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    ground_truth = np.array([3.0, 3.0, 2.0, 1.0, 1.0])

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 1.0)


def test_ndcg_minimal():

    # Set up data
    prediction = np.arange(10).astype(dtype=np.float32)
    ground_truth = np.flip(prediction, axis=0)

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 0.39253964576233569)


def test_ndcg_at_k():

    # Set up data
    prediction = np.array([0.3, 0.3, 0.2, 2.14, 0.23])
    ground_truth = np.array([3.0, 3.0, 2.0, 1.0, 1.0])

    # Compute and assert nDCG@3 value
    assert_equal(ndcg(prediction, ground_truth, k=3).data, 0.69031878315427031)


def test_empty_ndcg():

    # Set up data
    prediction = np.array([])
    ground_truth = np.array([])

    # Assert nDCG of empty lists
    assert_equal(ndcg(prediction, ground_truth).data, 0.0)


def test_ndcg_no_preferences():

    # Set up data
    prediction = np.array([0.3, 0.3, 0.2, 2.14, 0.23])
    ground_truth = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 1.0)


def test_ndcg_negative_predictions():

    # Set up data
    prediction = np.array([-0.1, -0.3, 1.9, -0.9, -0.2])
    ground_truth = np.array([0.0, 1.0, 1.0, 0.0, 0.0])

    # Compute and assert nDCG value
    assert_equal(ndcg(prediction, ground_truth).data, 0.8772153153380493)


@raises(ValueError)
def test_unequal_ndcg():

    # Set up data
    prediction = np.array([0.3, 0.3, 0.2])
    ground_truth = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 2.3])

    # This should raise a ValueError because the lists aren't of equal length
    ndcg(prediction, ground_truth)
