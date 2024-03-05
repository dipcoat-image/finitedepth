Development
===========

Testing
-------

To just test the module, install the package with ``test`` option and run
the following command in the project root directory::

   pytest

To perform the full test (e.g., linting and type checking), install with
``dev`` option and build the document before testing.
Then, run the following command in the project root directory::

   flake8
   black --check .
   isort --check .
   docformatter --check .
   doc8 .
   mypy src
   pytest
