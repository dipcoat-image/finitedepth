.. _config-reference:

Configuration file
==================

Configuration file consists of multiple entries representing individual
sets of analysis.

The following YAML file defines two entries:

.. code-block:: YAML

    data1:
        type: ...
        ...

    data2:
        type: ...
        ...

Each entry must have ``type`` field which specifies the analyzer.
It may have other fields required by the specified analyzer.

Refer to :func:`analyze_files` and :ref:`analyzer-reference` to know more about
analyzer.

