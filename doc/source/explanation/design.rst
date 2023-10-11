Core class design
=================

.. currentmodule:: finitedepth

This document describes the design principle of core classes:
:ref:`reference class <howto-reference>`,
:ref:`substrate class <howto-substrate>`,
:ref:`coating layer class <howto-coatinglayer>`,
:ref:`experiment class <howto-experiment>`, and
:ref:`analysis class <howto-analysis>`.

.. _generic:

Generic typing
--------------

.. _caching:

Caching
-------

Methods of image classes (:ref:`reference class <howto-reference>`,
:ref:`substrate class <howto-substrate>`,
:ref:`coating layer class <howto-coatinglayer>`) **MUST** be cached
in the instance itself, not in external container.

This is because external caching increases the reference count
of the instance, preventing it from being garbage collected on time.
The image classes consume large memory space because of image data.
If the objects stacks up in external cache, the program will soon crash.

It is generally OK to use external caching in
:ref:`experiment class <howto-experiment>` and
:ref:`analysis class <howto-analysis>`, as they need relatively small memory.
Of course, this does not hold if your implementations require
large memory space.

__slots__
---------

Classes in DipcoatImage-FiniteDepth intentionally do not define
:obj:`~object.__slots__`.

Slots are useful when large number of objects simultaneously exist.
This does not hold for core classes which are designed to be sequentially
constructed and destroyed.
Having slots will thus do more harm than good.

Verifying
---------
