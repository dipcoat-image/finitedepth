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

As described in :ref:`howto-runtime`, five abstract base classes consistitute
the analysis; :class:`ReferenceBase`, :class:`SubstrateBase`,
:class:`CoatingLayerBase`, :class:`ExperimentBase` and :class:`AnalysisBase`.

To define their concrete subclasses, there are some common rules to follow.

Do not modify signature
^^^^^^^^^^^^^^^^^^^^^^^

The abstract base classes strictly define their signatures which should not
be modified.

If you need to introduce new parameters, do not define them in ``__init__``
method. Instead, define dataclasses which wraps them and assign the dataclasses
as type variables which are passed to the constructor. This procedure is
explained in the following section.

Set type variables
^^^^^^^^^^^^^^^^^^

The abstract base classes are generic types whose type variables must
be set in concrete classes.

Each class defines different type variables. For example,
:class:`SubstrateBase` needs three type variables, each assigned to
:attr:`~SubstrateBase.Parameters`, :attr:`~SubstrateBase.DrawOptions`,
and :attr:`~SubstrateBase.Data`. On the other hand,
:class:`CoatingLayerBase` needs four type variables, each assigned to
:attr:`~CoatingLayerBase.Parameters`, :attr:`~CoatingLayerBase.DrawOptions`,
:attr:`~CoatingLayerBase.DecoOptions`, and :attr:`~CoatingLayerBase.Data`.
Concrete class must have all of its type variables assigned.

All type variables are dataclasses. To set the type variable,
take the following step:

#. Define dataclass.
#. Pass the dataclass as type variable of base class.
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
variables it defines. Read :ref:`howto-define-dataclass` to learn good
practices to define dataclass.

Cache by attribute
^^^^^^^^^^^^^^^^^^

Caching of methods **MUST** be done in the instance itself, not in
an external container.

**DO:**

.. code-block:: python

    class MySubstrate(SubstrateBase[...]):
        ...

        def foo_good(self):
            if not hasattr(self, "_foo"):
                self._foo = "bar"
            return self._foo

or equivalently,

.. code-block:: python

    from dipcoatimage.finitedepth.cache import attrcache

    class MySubstrate(SubstrateBase[...]):
        ...

        @attrcache("_foo")
        def foo_good(self):
            return "bar"

**DON'T:**

.. code-block:: python

    from functools import cache  # stores cache in external container

    class MySubstrate(SubstrateBase[...]):
        ...

        @cache
        def foo_bad(self):
            """DON'T DO THIS!"""
            return "bar"

This is because external caching increases the reference count
of *self*, preventing it from being garbage collected on time.
Our objects usually consume large memory space because of image
arrays; if the objects stacks up in external cache, the program will
soon crash.

It is generally OK to use external caching in experiment class and analysis
class, as they need relatively small memory space. Of course, this does not
hold if your implementations require large memory space.

Implement abstract methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

Concrete class needs to implement every abstract method in abstrac base class.

You can find the abstract methods in :ref:`api` page. There are some common
methods that deserve mentioning here, though.

#. ``verify()``
    Defined in: :class:`ReferenceBase`, :class:`SubstrateBase`,
    :class:`CoatingLayerBase`, :class:`ExperimentBase`, :class:`AnalysisBase`.

    Checks the instance parameters and raises error before running expensive
    analysis.

#. ``analyze()``
    Defined in: :class:`ReferenceBase`, :class:`SubstrateBase`,
    :class:`CoatingLayerBase`.

    Return a dataclass which contains analysis result. Type of the dataclass
    must be ``Data`` attribute. Analysis algorithm should be implemented here.

#. ``draw()``
    Defined in: :class:`ReferenceBase`, :class:`SubstrateBase`,
    :class:`CoatingLayerBase`.

    Return visualization result in RGB. This method must return in any case.
    If analysis fails and cannot be visualized, it must at least return
    original image in RGB.

Examples
--------

Which class should I implement?
-------------------------------

Defining configuration class
----------------------------
