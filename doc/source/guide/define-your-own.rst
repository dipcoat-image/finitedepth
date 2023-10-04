.. _howto-define-class:

How to define your own class
============================

.. currentmodule:: finitedepth

In this document, you will learn how to implement your own analysis program
by defining custom classes.

DipcoatImage-FiniteDepth provides extensible API with abstract base classes.
Classes such as :class:`Reference` or :class:`RectSubstrate` are their
concrete implementations. By following the same API, you can implement your
own classes which seamlessly bind with the framework.

Basic rules
-----------

As described in :ref:`howto-runtime`, there are five abstract base classes
which consistitute the analysis:

#. :class:`ReferenceBase`
#. :class:`SubstrateBase`
#. :class:`CoatingLayerBase`
#. :class:`ExperimentBase`
#. :class:`AnalysisBase`

To define their concrete subclasses, there are some common rules to follow.

Do not modify signature
^^^^^^^^^^^^^^^^^^^^^^^

The abstract base classes strictly define their signatures which should not
be modified.

If you need to introduce new parameters, do not define them in ``__init__``
method. Instead, wrap them with dataclasses with are passed to the class
constructor. This procedure is explained in the following section.

Set type variables
^^^^^^^^^^^^^^^^^^

The abstract base classes are generic types whose type variables must
be set in concrete classes.

Each class defines different type variables. For example,
:class:`SubstrateBase` has three type variables, each assigned to
:attr:`~SubstrateBase.Parameters`, :attr:`~SubstrateBase.DrawOptions`,
and :attr:`~SubstrateBase.Data`. On the other hand,
:class:`CoatingLayer` has four type variables, each assigned to
:attr:`~CoatingLayer.Parameters`, :attr:`~CoatingLayer.DrawOptions`,
:attr:`~CoatingLayer.DecoOptions`, and :attr:`~CoatingLayer.Data`.
Concrete class must have all of its type variables assigned.

All type variables are dataclasses. To set the type variable,
take the following step:

#. Define the dataclass.
#. Pass the dataclass as the type variable of base class.
#. Assign the dataclass as the type variable attribute.

The following is the example of setting type variable for custom
substrate class.

.. code-block:: python

    @dataclass
    class MyParameters:
        ...

    @dataclass
    class MyDrawOptions:
        ...

    @dataclass
    class MyData:
        ...

    class MySubstrate(SubstrateBase[MyParameters, MyDrawOptions, MyData]):
        Parameters = MyParameter
        DrawOptions = MyDrawOptions
        Data = MyData

See :ref:`api` of each abstract base class for the list of type
variables it defines. Read :ref:`howto-define-dataclass` for good
practice to define dataclass.

Set slots
^^^^^^^^^

Abstract base classes define class attribute :obj:`object.__slots__`
to save memory space.

When your concrete instance needs to define new attributes, you must
declare them in ``__slots__``. This is especially important when you
need to cache your method.

Cache by attribute
^^^^^^^^^^^^^^^^^^

Caching of methods **MUST** be done by private attribute.

**DO:** 

.. code-block:: python

    class MySubstrate(SubstrateBase[...]):
        ...
        __slots__ = ("_foo",)

        def foo_good(self):
            if not hasattr(self, "_foo"):
                self._foo = "bar"
            return self._foo

**DON'T:**

.. code-block:: python

    from functools import cache

    class MySubstrate(SubstrateBase[...]):
        ...

        @cache
        def foo_bad(self):
            """DON'T DO THIS!"""
            return "bar"

This is because external caching increases the reference count
of *self*, preventing it from being garbage collected in time.
Our objects usually consume large memory because of image
arrays. If the objects linger in cache stack, the program will
crash after constructing a few instances.

It is generally OK to use external caching in experiment class
and analysis class, as they need relatively small memory space.

Implement abstract methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

Examples
--------

Which class should I implement?
-------------------------------

Defining configuration class
----------------------------
