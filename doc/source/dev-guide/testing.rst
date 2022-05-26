=======
Testing
=======

This document explains how to run tests while developing.

Although CI workflow runs all the tests mentioned here, it is good to run them in your environment before pushing the commits.

Installing
==========

Running test requires additional packages to be installed.
As mentioned in :ref:`Installation <install-options>` document, you can install DipCoatImage-FiniteDepth with ``test`` option to install these packages altogether.

If you wish to run tests in headless environment, install with ``test-ci`` option instead.

Linting
=======

DipCoatImage-FiniteDepth use `Flake8 <https://flake8.pycqa.org/en/latest/>`_ and `Black <https://black.readthedocs.io/en/stable/>`_ for linting.

One of the few inconsistencies between two modules is line length policy.
Flake8 encourages 79 characters for code lines and Black prefers 88 characters.
DipCoatImage-FiniteDepth follows Blacks chooses 88 limit.

Flake8 limits the line length for docstrings and comments to 72 characters, 7 characters smaller than code line.
Black does not have any explicit policy for this, so to respect Flake8 we limit these lines to 81 characters.

Flake8
======

To check code quality with Flake8, simpliy run the following command in root path (where ``setup.py`` file is located).

.. code-block:: bash

   $ flake8

Black
=====

The following commands lints and modifies your code using Black.

.. code-block:: bash

   $ black .

To only check the codes without modifying them, run the following command.

.. code-block:: bash

   $ black --check .

Type Checking
=============

DipCoatImage-FiniteDepth use `Mypy <https://mypy.readthedocs.io/en/stable/>`_ for type check.
Run the following command to check your code.

.. code-block:: bash

   $ mypy .

.. note::

   `mypy == 9.6.0` has issue for checking `TypeVar[Protocol]`. Use other version.

Unit testing
============

DipCoatImage-FiniteDepth use `pytest <https://docs.pytest.org/en/stable/>`_ for unit testing.
Run the following command to test your entire code.

.. code-block:: bash

   $ pytest

Building document
=================

DipCoatImage-FiniteDepth use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to build documentation.
The following command changes your current directory to ``doc/`` and tries building the files.

.. code-block:: bash

   $ make html SPHINXOPTS="-W --keep-going"
