import numpy as np
from nose.tools import assert_equal, assert_true
from shoelace.functions.logcumsumexp import logcumsumexp, LogCumsumExp
from chainer import Variable


def test_forward():

    # Construct test data
    x = Variable(np.array([5., 3., 3., 1., 0.]))
    expected_result = Variable(np.array([5.259069729874858, 3.7816718231374824,
                                         3.1698460195562856, 1.313261687518223,
                                         0.]))

    # Run forward pass with shorthand notation
    result = logcumsumexp(x)

    # Assert that the result equals the expected result
    assert_true(np.array_equal(result.data, expected_result.data))


def test_backward():

    # Construct test data
    x = Variable(np.array([5., 3., 3., 1., 0.]))
    g = Variable(np.ones(5))
    expected_result = np.array([0.7717692057972512, 0.562087881852882,
                                1.4058826163342215, 0.9213241007090265,
                                1.3389361953066183])

    # Generate object
    lcse = LogCumsumExp()

    # Run forward and backward pass
    lcse.forward((x.data,))
    result = lcse.backward((x.data, ), (g.data, ))

    # Assert that the result equals the expected result
    assert_true(np.array_equal(result[0], expected_result))
