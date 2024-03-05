Getting started
===============

Installation
------------

DipCoatImage-FiniteDepth can be downloaded from
`PyPI <https://pypi.org/project/dipcoatimage-finitedepth/>`_ by using :mod:`pip`::

   pip install dipcoatimage-finitedepth

You can also install with optional dependencies as::

   pip install dipcoatimage-finitedepth[dev]

Available optional dependencies for DipCoatImage-FiniteDepth are:

* ``test``: run tests.
* ``doc``: build documentations.
* ``dev``: every dependency (for development).

Usage
-----

Analysis starts with writing a :ref:`configuration file <config-reference>`,
either as YAML or JSON.

.. code-block:: yaml

   data1:
    type: CoatingImage
    referencePath: ref.png
    targetPath: target.png
    output:
        layerData: output/data.csv
   data2:
      type: MyType
      my-parameters: ...
   data3:
      ...

The ``type`` field is important.
It defines how the analysis is done and what parameters are required.
You can define and register your own type by writing a :ref:`plugin <plugin>`.

After specifying the parameters in configuration file, pass it to
:ref:`'finitedepth analyze' <command-reference>` command to perform analysis::

   finitedepth analyze config.yml

Refer to :ref:`tutorial` page for a runnable example.
