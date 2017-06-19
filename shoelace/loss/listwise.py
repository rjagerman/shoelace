import numpy as np
import chainer
import chainer.functions as F
from chainer import Chain, cuda
from shoelace.functions.logcumsumexp import logcumsumexp


class AbstractListLoss(Chain):
    """
    An abstract listwise loss function
    
    This loss calls the prediction function on the target variable and calls
    a local `AbstractListLoss.loss` function which should be implemented by
    subclasses
    """
    def __init__(self, predictor):
        super(AbstractListLoss, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        x_hat = self.predictor(x)
        loss = self.loss(x_hat, t)
        return loss

    def loss(self, x, t):
        raise NotImplementedError


class ListMLELoss(AbstractListLoss):
    """
    The ListMLE loss as in Xia et al (2008), Listwise Approach to Learning to
    Rank - Theory and Algorithm.
    """
    def __init__(self, predictor):
        super(ListMLELoss, self).__init__(predictor=predictor)

    def loss(self, x, t):
        """
        Assuming target labels are already sorted by relevance
        :param x: The x variable 
        :param t: The target variable
        :return: The loss
        """
        final = logcumsumexp(x)
        return F.sum(final - x)


class ListNetLoss(AbstractListLoss):
    """
    The Top-1 approximated ListNet loss as in Cao et al (2006), Learning to
    Rank: From Pairwise Approach to Listwise Approach
    """
    def __init__(self, predictor):
        super(ListNetLoss, self).__init__(predictor=predictor)

    def loss(self, x, t):
        """
        ListNet top-1 reduces to a softmax and simple cross entropy
        :param x: The x variable
        :param t: The target variable
        :return: The loss
        """
        st = F.softmax(t, axis=0)
        sx = F.softmax(x, axis=0)
        return -F.mean(st * F.log(sx))


class ListPLLoss(AbstractListLoss):
    """
    The ListPL loss, a stochastic variant of ListMLE that in expectation
    approximates the true ListNet loss.
    """
    def __init__(self, predictor, α=15.0):
        super(ListPLLoss, self).__init__(predictor=predictor)
        self.α = α

    def loss(self, x, t):
        # Sample permutation from PL(t)
        index = self.pl_sample(t)
        x = x[index]

        # Compute MLE loss
        final = logcumsumexp(x)
        return F.sum(final - x)

    def pl_sample(self, t):
        """
        Sample from the plackett luce distribution directly

        :param t: The target labels 
        :return: A random permutation from the plackett-luce distribution
                 parameterized by the target labels
        """
        xp = cuda.get_array_module(t)
        if not hasattr(xp, 'asnumpy'):
            xp.asnumpy = lambda x: x
        t = t[:, 0]

        probs = xp.exp(t * self.α)
        probs /= xp.sum(probs)
        return np.random.choice(probs.shape[0], probs.shape[0], replace=False,
                                p=xp.asnumpy(probs))

