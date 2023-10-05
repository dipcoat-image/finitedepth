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
:class:`SubstrateBase` has four type variables, each specifying
the type of the reference instance it takes, its parameters, its drawing
options, and its analysis data. The latter three are assigned to
:attr:`~SubstrateBase.Parameters`, :attr:`~SubstrateBase.DrawOptions`,
and :attr:`~SubstrateBase.Data`. On the other hand,
:class:`CoatingLayerBase` has five type variables; the substrate type
and the four types assigned to :attr:`~CoatingLayerBase.Parameters`,
:attr:`~CoatingLayerBase.DrawOptions`, :attr:`~CoatingLayerBase.DecoOptions`,
and :attr:`~CoatingLayerBase.Data`.

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

    class MySubstrate(SubstrateBase[ReferenceBase, MyParameters, MyDrawOptions, MyData]):
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

When and which to implement?
----------------------------

:class:`ReferenceBase` is not something that you usually want to implement.
As it is merely a wrapper, :class:`Reference` will almost always be enough.
That being said, a possible scenario is that you want to apply machine learning
to automatically determine optimal ROIs.

:class:`SubstrateBase` and :class:`CoatingLayerBaseBase` are the APIs you will
frequently implement. If you want to acquire geometry-specific data, you need
to define both the substrate class and the coating layer class. For example,
:class:`RectSubstrate` and :class:`RectLayerShape` are designed to analyze the
coating layer over rectangular substrate.

:class:`ExperimentBase` is again unlikely to be implemented. If you need
specific way to construct the coating layer instances, define your experiment
class. For example, you may have ground truth data of the substrate location in
target image, so you need your experiment instance to read the data and pass it
to the constructor of coating layer class.

:class:`AnalysisBase` may be implemented to for different file IO API.
Instead of :mod:`cv2` and :mod:`Pillow` library that :class:`Analysis` relies
on, your implementation can perhaps directly open ``ffmpeg`` subprocess.

If you design multiple classes to cooperate, you may want to assign one class
as the type variable of another. For example, :class:`RectSubstrate` is
assigned to the ``SubstrateType`` variable of :class:`RectLayerShape`,
informing type checkers(e.g., :mod:`mypy`) that any other substrate should not
be passed.

Defining configuration class
----------------------------
