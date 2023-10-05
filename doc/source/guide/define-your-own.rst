.. _howto-define-class:

How to define your own class
============================

.. currentmodule:: finitedepth

In this document, you will learn how to implement your own analysis program
by defining custom classes.

DipcoatImage-FiniteDepth provides extensible API with abstract base classes.
Classes such as :class:`Reference` or :class:`RectSubstrate` are their
concrete subclasses. By following the same API, you can implement your
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

#. Define dataclasses.
#. Set type variable of base class with the dataclasses.
#. Assign the dataclasses as the type variable attribute.

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

Read :ref:`caching` page for explanation.

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

:class:`SubstrateBase` and :class:`CoatingLayerBase` are the APIs you will
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
Instead of relying on :mod:`cv2` and :mod:`PIL` libraries as :class:`Analysis`
do, your implementation can perhaps directly open ``ffmpeg`` subprocess to
write video.

If you design multiple classes to cooperate, you may want to assign one class
as the type variable of another. For example, :class:`RectSubstrate` is
assigned to the ``SubstrateType`` variable of :class:`RectLayerShape`,
informing type checkers(e.g., :mod:`mypy`) that any other substrate should not
be passed.

Subclassing strategy
--------------------

Again, your goal is to define a concrete class which inherits abstract base
class. There are three approaches to achieve this:

#. Full implementation
    Directly inherit the abstract base class to final class.

    This is the standard approach which has been explained so far.

#. Partial implementation
    Define partially-implemented base class first, which final class inherits.

    This is useful when your implementation has abstract features that can be
    generalized. For example, :class:`RectSubstrate` inherits
    :class:`PolySubstrateBase` which is an abstract subclass of
    :class:`SubstrateBase`. By having this additional abstract layer,
    various polygonal substrate classes can be easily implemented.

#. Re-implementation
    Subclass concrete class.

    You can inherit already-concrete class to redefine some of its methods.
    This approach can be useful if you just want to make minor modifications.

Defining configuration class
----------------------------

You can also implement your own configuration class other than :class:`Config`
by subclassing :class:`ConfigBase`. Configuration class is just a dataclass
with a few abstract methods, so all you need is just implement them.

You may want to define your own configuration class in limited occasions where
you need different file IO API. Instead of relying on :mod:`cv2` and :mod:`PIL`
libraries as :class:`Config` do, your implementation can perhaps directly open
``ffmpeg`` subprocess to read video.

Note that you cannot use your custom configuration class in
:ref:`command-line analysis <basic-example>`. Instead, you can use yours in
runtime or write a simple script to invoke it.

You should also note that you cannot change the data fields and
:ref:`configuration file structure <config-reference>` by subclassing
:class:`ConfigBase`. If you want different structure, define your own class
from scratch.
