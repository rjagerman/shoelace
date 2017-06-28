.. _getting_started-ref:

=====================
Getting Started Guide
=====================

In this guide we will go through a simple example from start to finish. You
will learn how to load data, create an iterator, setup a neural network and
learn its parameters using a ranking loss function.

Loading a data set
==================
We will load a very simple example data set in RankSVM format.

.. code-block:: python

    from shoelace.dataset import LtrDataset

    with open('./dataset.txt', 'r') as file:
        dataset = LtrDataset.load_txt(file)

Chainer works with a concept of iterators that feed a neural network with
batches of data. We need to load our data into a Learning to Rank iterator to
start using it:

.. code-block:: python

    from shoelace.dataset import LtrIterator

    iterator = LtrIterator(dataset, repeat=True, shuffle=True)

You can find more detailed information about the data set loading facilities
that Shoelace provides on the :doc:`/datasets` section of the documentation.

Setting up a network
====================
For our simple example we will set up a single-layer linear neural network. This
is equivalent to a linear function of the input features.

.. code-block:: python

    from chainer import links
    predictor = links.Linear(None, 1)

We encourage the reader to experiment with a wide variety of neural
architectures. For more information about designing different architectures we
refer you to the `documentation of Chainer <https://docs.chainer.org>`_.

Choosing a loss function
========================
Shoelace currently provides 3 different list-wise loss functions. For this guide
we will use the ListNet loss (top-1 approximation):

.. code-block:: python

    from shoelace.loss.listwise import ListNetLoss
    loss = ListNetLoss(predictor)


Training and evaluation
=======================
We now have all the pieces set up to start training our network. What follows is
standard Chainer code for setting up an optimizer, updater and trainer. There
are many options and choices to be made here, but they fall outside the scope of
this guide. You will be able to find much more information about optimizing the
network on the `documentation of Chainer <https://docs.chainer.org>`_.

.. code-block:: python

    from chainer import training, optimizers
    from chainer.training import extensions

    # Build optimizer, updater and trainer
    optimizer = optimizers.Adam()
    optimizer.setup(loss)
    updater = training.StandardUpdater(iterator, optimizer)
    trainer = training.Trainer(updater, (40, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    # Train neural network
    trainer.run()

