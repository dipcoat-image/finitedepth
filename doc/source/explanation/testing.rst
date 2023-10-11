Testing
=======

This document explains how to run tests.

Prerequisites
-------------

Install DipCoatImage-FiniteDepth with ``dev`` option.
Refer to :ref:`install` page.

Before testing, you must build the document first to generate the
referenced files.
Refer to :ref:`document` page.

Testing commands
----------------

In the project root directory, run the following commands.

.. code-block:: bash

   flake8
   black --check .
   isort --check .
   docformatter --check .
   doc8 .
   mypy src
   pytest
