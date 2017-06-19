import numpy as np
from nose.tools import raises, assert_equal, assert_almost_equal
from chainer import Variable, Link
from shoelace.loss.listwise import ListNetLoss, ListMLELoss, ListPLLoss, \
    AbstractListLoss


class Identity(Link):
    def __init__(self):
        super(Identity, self).__init__()

    def __call__(self, x):
        return x


@raises(NotImplementedError)
def test_abstract_notimplemented():
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[3., 2., 0., 1.]]).T
    loss = AbstractListLoss(Identity())
    loss(x, t)


def test_listnet():
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[0.5, 1.0, 0.3, 0.5]]).T
    loss = ListNetLoss(Identity())

    result = loss(x, t)
    assert_equal(result.data, 0.43439806229182915)


def test_listnet_near_zero_loss():
    x = np.array([[600., 400., 200., 0.]]).T
    t = np.array([[600., 400., 200., 0.]]).T
    loss = ListNetLoss(Identity())

    result = loss(x, t)
    assert_almost_equal(result.data, 0.0)


def test_listmle():
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[0.5, 1.0, 0.3, 0.5]]).T
    loss = ListMLELoss(Identity())

    result = loss(x, t)
    assert_equal(result.data, 1.3587430505329978)


def test_listmle_near_zero_loss():
    x = np.array([[600., 400., 200., 0.]]).T
    t = np.array([[600., 400., 200., 0.]]).T
    loss = ListMLELoss(Identity())

    result = loss(x, t)
    assert_almost_equal(result.data, 0.0)


def test_listpl():
    np.random.seed(4101)
    x = np.array([[3., 3., 2., 0.]]).T
    t = np.array([[0.5, 1.0, 0.3, 0.5]]).T
    loss = ListPLLoss(Identity())

    result = loss(x, t)
    assert_equal(result.data, 3.358743050532998)


def test_listpl_near_zero_loss():
    np.random.seed(4101)
    x = np.array([[40., 20., 0.]]).T
    t = np.array([[40., 20., 0.]]).T
    loss = ListPLLoss(Identity())

    result = loss(x, t)
    assert_almost_equal(result.data, 0.0)
