.. _plugin:

Writing plugins
===============

.. currentmodule:: finitedepth

By writing a plugin, you can define your own analysis routines and specify them
in your configuration file.

.. note::

    If you are not familiar with plugins and entrypoints, you may want to read
    `Python packaging guide <https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/>`_
    and `Setuptools documentation <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_
    first.

Analyzers
---------

Let us suppose that we want to implement an analysis type ``Foo`` for the
following configuration file entry:

.. code-block:: yaml

    foo:
        type: Foo
        ...

For this, we can write a simple plugin named ``finitedepth-foo``.
The package structure will be::

    finitedepth-foo
    ├── pyproject.toml
    └── foo.py

In ``foo.py`` we define:

.. code-block:: python

    def foo_analyzer(name: str, data: dict):
        ... # do whatever you want

The signature of analyzers is specified in :func:`analyze_files`.
Then, in ``pyproject.toml`` we define a table to register
:func:`foo_analyzer` to ``Foo``:

.. code-block:: toml

    [project.entry-points."finitedepth.analyzers"]
    Foo = "foo:foo_analyzer"

Now, by installing the ``finitedepth-foo`` package, the analysis type ``Foo``
will be recognized.

Samples
-------

As shown in :ref:`tutorial`, DipCoatImage-FiniteDepth supports
:ref:`'finitedepth samples' <command-reference>` command to print the path
to the directory where sample files are stored. If your plugin has its own
sample directory, you can register it to the same API.

To distribute the sample files as package data, the ``finitedepth-foo``
needs to be a little more complicated::

    finitedepth-foo
    ├── pyproject.toml
    ├── MANIFEST.in
    └── src
        └── foo
            ├── samples
            └── __init__.py

Make sure that the sample directory is included in ``MANIFEST.in``.
Then, in ``__init__.py`` we define:

.. code-block:: python

    from importlib.resources import files

    def sample_path():
        return str(files("finitedepth-foo").joinpath("samples"))

And in ``pyproject.toml`` we define a table:

.. code-block:: toml

    [project.entry-points."finitedepth.samples"]
    foo = "foo:sample_path"

Then, invoking ``finitedepth samples foo`` will print the path to
the samples directory.
Note that :func:`sample_path` can have different signature as long as it
returns the correct path when called with empty argument.
