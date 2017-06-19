import numpy as np
from chainer import training, optimizers, links
from chainer.dataset import convert
from nose.tools import assert_almost_equal

from shoelace.iterator import LtrIterator
from shoelace.loss.listwise import ListNetLoss
from test.utils import get_dataset


def test_linear_network():

    # To ensure repeatability of experiments
    np.random.seed(1042)

    # Load data set
    dataset = get_dataset(True)
    iterator = LtrIterator(dataset, repeat=True, shuffle=True)
    eval_iterator = LtrIterator(dataset, repeat=False, shuffle=False)

    # Create neural network with chainer and apply our loss function
    predictor = links.Linear(None, 1)
    loss = ListNetLoss(predictor)

    # Build optimizer, updater and trainer
    optimizer = optimizers.Adam(alpha=0.2)
    optimizer.setup(loss)
    updater = training.StandardUpdater(iterator, optimizer)
    trainer = training.Trainer(updater, (10, 'epoch'))

    # Evaluate loss before training
    before_loss = eval(loss, eval_iterator)

    # Train neural network
    trainer.run()

    # Evaluate loss after training
    after_loss = eval(loss, eval_iterator)

    # Assert precomputed values
    assert_almost_equal(before_loss, 0.26958397)
    assert_almost_equal(after_loss, 0.2326711)


def eval(loss_function, iterator):
    """
    Evaluates the mean of given loss function over the entire batch in given
    iterator
    
    :param loss_function: The loss function to evaluate 
    :param iterator: The iterator over the evaluation data set
    :return: The mean loss value
    """
    iterator.reset()
    results = []
    for batch in iterator:
        input_args = convert.concat_examples(batch)
        results.append(loss_function(*input_args).data)
    return np.mean(results)
