.. _datasets-ref:

========
Datasets
========

Datasets in Shoelace can be loaded either in RankSVM .txt format or a custom
binary format. When you are just getting started, you will want to load data
sets using RankSVM format as most data sets are available in this format.
However, for future reference it is much faster to save and load the custom
binary format.

RankSVM Format
==============
The most typical format to load a Learning to Rank data set is using RankSVM
format. Many publicly available anotated data sets have used this format. Given
a Learning to Rank data set called `dataset.txt`, you can load it as follows:

.. code-block:: python

    from shoelace.dataset import LtrDataset

    with open('./dataset.txt', 'r') as file:
        dataset = LtrDataset.load_txt(file)

Some data sets in RankSVM format do not come with query-level normalization. For
most loss functions that depend on exponentials it is very recommended to do
query-level normalization to prevent overflow errors. Fortunately, Shoelace has
normalization built-in as part of the data set loading facilities. You can load
a data set with query-level normalization by setting the `normalize` parameter
to `True`:

.. code-block:: python

    with open('./dataset.txt', 'r') as file:
        dataset = LtrDataset.load_txt(file, normalize=True)



Binary Format
=============
Shoelace provides a custom binary format that is much faster to save and load
than the RankSVM text format. Once you have loaded a dataset, you can save it
to a binary file once, so that you can load it much faster in future
experiments:

.. code-block:: python

    with open('./dataset.bin', 'wb') as file:
        dataset.save(file)

Once saved, you can load the file again. It will be several orders of magnitude
faster than loading the corresponding txt file:

.. code-block:: python

    with open('./dataset.bin', 'rb') as file:
        dataset = LtrDataset.load(file)


Iterators
=========
Chainer works with a concept of iterators that feed a neural network with
batches of data. We provide an iterator specifically designed for Learning to
Rank tasks. Such an iterator considers a single query (and all associated
query-document pairs) as a single learning batch. This will allow us to do
list-wise Learning to Rank by taking into account the entire ranked list for a
given query. As a consequence, not every minibatch will have the same size,
which is fortunately no problem for Chainer's run-time architecture.

You can load the data set into an iterator as follows:

.. code-block:: python

    from shoelace.dataset import LtrIterator

    iterator = LtrIterator(dataset)

You can additionally specify whether the iterator should repeat forever (e.g.
for training data, but not for test data) and whether the order of data should
be shuffled on every epoch:

.. code-block:: python

    iterator = LtrIterator(dataset, repeat=True, shuffle=True)
