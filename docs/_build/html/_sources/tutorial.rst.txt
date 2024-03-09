Tutorial
#########

Getting Started
===============

Install the 'bom_analyzer' package from PYPI using:

.. code-block::

    pip install bom_analyzer

After installing, import the library into your file using:

.. code-block:: python

    import bom_analyzer as ba

Using the Library
==================

.. note::

    The input dataset must be a csv with the following attributes:

    .. list-table::

        * - SERNUM
          - Column denoting the serial number of the product
        * - HWRMA
          - Column denoting whether a product has a known error with value True or False
        * - CPN_#
          - Every component must have an associated CPN number

To begin, call `run_sentence_transform` with the path of the product dataset.
To use hardware acceleration, make sure to specify the "device" attribute.
It is also recommended, that an archive path is specified to store the output.

.. code-block:: python

    embeddings = ba.run_sentence_transform('dataset.csv', device='cuda', archive_path='embeddings.npy')

Next, with the output embeddings, call `run_optimizer` to determine the hyperparameters.
that groups the data most effectively (based on DBCV score).
The number of trials, random seed, and a location to archive the output can all be specified as shown below.

.. code-block:: python

    params = ba.run_optimizer(embeddings, seed=42, trials=50, archive_path='params.json')

Now that you have the embeddings and the ideal hyperparameters, call `run_dimension_reduction`
to convert the embeddings into two-dimensional data.

This function accepts a path to, or dataframe of the original dataset and
appends the result in the columns `DATA_X` and `DATA_Y`.

.. code-block:: python

    table = ba.run_dimension_reduction(table, embeddings, params, seed=42, archive_path='dataset.csv')

With the dimension reduced data now accessible, pass the updated table into `run_clustering`
to get the cluster labels associated with each product.

As before, this function accepts a path to, or dataframe of the original dataset
with `DATA_X` and `DATA_Y` already defined. It then appends the result in the column `CLUSTERS`.

.. code-block:: python

    table = ba.run_clustering(table, params, archive_path='dataset.csv')

Now, we can determine the density of errors in these groups with the `label_outliers` function.

As before, this function accepts a path to, or dataframe of the original dataset
with `DATA_X`, `DATA_Y`, and `CLUSTERS` already defined.
It then appends the result in the column `OUTLIER_DENSITY`.

.. code-block:: python

    table = ba.label_outliers(table, archive_path='dataset.csv')

Last, we can find the components associated with these errors using the `report_suspect_components` function.


As before, this function accepts a path to, or dataframe of the original dataset
with `DATA_X`, `DATA_Y`, `CLUSTERS`, and `OUTLIER_DENSITY` already defined.

The function returns a dataframe of components unique to the 'n' clusters with the highest
density of errors. This dataframe will contain columns `CPN`, `DateCode`, `LOTCODE`, `MPN`, and `RD`.

.. code-block:: python

    components = ba.report_suspect_components(table, num_clusters=n, archive_path='components.csv')

