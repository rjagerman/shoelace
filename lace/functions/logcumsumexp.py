from chainer import cuda
from chainer import function
from chainer.utils import type_check


class LogCumsumExp(function.Function):

    def __init__(self, axis=None):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(isinstance(a, int) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

        if self.axis is not None:
            for axis in self.axis:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        x, = inputs
        m = x.max(axis=self.axis, keepdims=True)
        y = x - m
        xp.exp(y, out=y)
        y_sum = xp.flip(xp.cumsum(xp.flip(y, axis=0)), axis=0)
        self.y = xp.transpose(xp.asarray(xp.log(y_sum) + m))
        return self.y,

    def backward(self, inputs, grads):
        xp = cuda.get_array_module(*inputs)
        x, = inputs
        gy, = grads

        y = self.y
        gx = gy * xp.exp(x) * xp.cumsum(xp.exp(-y), axis=0)
        return gx,


def logcumsumexp(x):
    """Log-sum-exp of array elements over a given axis.

    This function calculates logarithm of sum of exponential of array elements.

    .. math::

       y_i = \\log\\left(\\sum_j \\exp(x_{ij})\\right)

    Args:
        x (~chainer.Variable): Elements to log-sum-exp.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return LogCumsumExp(axis=0)(x)
