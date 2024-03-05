.. currentmodule:: finitedepth

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

Caching
-------

Image processing is computationally expensive. To enhance the performance,
expensive functions that are called multiple times can be cached.

When caching methods of image classes (:class:`ReferenceBase`,
:class:`SubstrateBase`, and :class:`CoatingLayerBase`), remember that the
data **MUST** be stored in the instance itself, not in external container,
e.g., using :func:`functools.cache`.
This is because external caching increases the reference count of the
instance, preventing it from being garbage collected on time. Instances
storing image consume large memory space, and will crash the program
if they are keep being kept in external cache.

This also requires that attributes of these classes should not be mutated.
Using ``@property`` is recommended to prevent accidental mutation.

__slots__
----------

Image classes (:class:`ReferenceBase`, :class:`SubstrateBase`, and
:class:`CoatingLayerBase`) do not need to define :obj:`object.__slots__`.

Slots are useful when large number of objects simultaneously exist. This
does not hold for core classes which are designed to be sequentially
constructed and destroyed.
