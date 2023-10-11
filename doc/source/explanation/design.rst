Core class design
=================

.. _generic:

Generic typing
--------------

.. _caching:

Caching
-------

Caching of methods **MUST** be done in the instance itself, not in
an external container.

This is because external caching increases the reference count
of *self*, preventing it from being garbage collected on time.
Our objects usually consume large memory space because of image
arrays; if the objects stacks up in external cache, the program will
soon crash.

It is generally OK to use external caching in experiment class and analysis
class, as they need relatively small memory space. Of course, this does not
hold if your implementations require large memory space.

__slots__
---------

Classes in DipcoatImage-FiniteDepth intentionally not define
:obj:`~object.__slots__`.

Slots are useful when large number of objects simultaneously exist,
which is unlikely in our case where each object consumes large memory
because of the image data it stores. Instead of constructing multiple
objects, we need to handle small number of objects and destroy them
as soon as possible. Hence, having slots does more harm than good.

Verifying
---------
