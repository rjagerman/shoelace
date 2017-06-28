import numpy as np
from nose.tools import assert_equal, assert_almost_equal
from shoelace.loss.listwise import listnet, listmle, listpl


def test_listnet():
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[0.5, 1.0, 0.3, 0.5]]).T

    result = listnet(x, t)
    assert_equal(result.data, 0.43439806229182915)


def test_listnet_near_zero_loss():
    x = np.array([[600., 400., 200., 0.]]).T
    t = np.array([[600., 400., 200., 0.]]).T

    result = listnet(x, t)
    assert_almost_equal(result.data, 0.0)


def test_listmle():
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[0.5, 1.0, 0.3, 0.5]]).T

    result = listmle(x, t)
    assert_equal(result.data, 4.545076727008247)


def test_listmle_near_zero_loss():
    x = np.array([[600., 400., 200., 0.]]).T
    t = np.array([[600., 400., 200., 0.]]).T

    result = listmle(x, t)
    assert_almost_equal(result.data, 0.0)


def test_listpl():
    np.random.seed(4101)
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[0.5, 1.0, 0.3, 0.5]]).T

    result = listpl(x, t)
    assert_equal(result.data, 3.358743050532998)


def test_listpl_near_zero_loss():
    np.random.seed(4101)
    x = np.array([[40., 20., 0.]]).T
    t = np.array([[40., 20., 0.]]).T

    result = listpl(x, t)
    assert_almost_equal(result.data, 0.0)
