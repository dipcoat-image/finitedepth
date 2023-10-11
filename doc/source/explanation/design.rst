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

Core classes are generic types which support static type checking
by :mod:`mypy`.

In :ref:`howto-define-class` page, type variables are passed to
abstract base classes, e.g.,
``SubstrateBase[ReferenceBase, MyParameters, MyDrawOptions, MyData]``
for subclassing. This type annotation informs the type checker of
what types are allowed.

What is type variable?
^^^^^^^^^^^^^^^^^^^^^^

Consider an identity function ``f(x)`` which takes integer argument.

.. code-block:: python

    def f(x: int) -> int:
        return x

The type annotation indicates that an integer goes in,
and an integer comes out.

What if the argument can be integer or string? This is where we need
type variable.

.. code-block:: python

    from typing import TypeVar

    T = TypeVar("T", bound=int | str)

    def f(x: T) -> T:
        return x

This annotation indicates that the input type and the output type are same.
It can be mathematically written as :math:`f: T \rightarrow T`.

What is generic type?
^^^^^^^^^^^^^^^^^^^^^

Generic type has type parameter, which can be later specified.

.. code-block:: python

    from typing import TypeVar, Generic

    T = TypeVar("T", bound=int | str)

    class G(Generic[T]):
        def __init__(self, x: T):
            self._x = x
        def x(self) -> T:
            return self._x

    G[int](1)  # OK
    G[str](1)  # Violate! (incompatible variable)
    G[float](1.0)  # Violate! (invalid type variable)

Here, ``G`` is a generic type which takes integer or string.
We annotated that ``G.x()`` returns the same type as its parameter.

``G`` itself has free type variable ``T``, which can be substituted by
:obj:`int` or :obj:`str`. ``G[int]`` indicates that ``T`` is fixed to
:obj:`int`, therefore the parameter must be an integer. ``G[str]`` is
also allowed, but ``G[float]`` is illegal since ``T`` is bound to
``int | str``.

Once you substituted every type variable, the class is no longer generic.
Let us define a subclass of ``G[int]``:

.. code-block:: python

    class H(G[int]):
        ...

    H[int](1)  # Error! (H is not generic)
    H(1)  # OK
    H("foo")  # Violate! (incompatible variable)

Here, ``H[int]`` does not pass type check because ``H`` no longer has
free type variable. In fact, it cannot even run in runtime.
``H("foo")`` is allowed in runtime, but it violates type annotation
since the parameter of ``H`` is fixed to ``int`` by subclassing ``G[int]``.

Core class as generic type
^^^^^^^^^^^^^^^^^^^^^^^^^^

Core classes define various type variables, which are substituted by
subclasses. For example, :class:`CoatingLayerBase` defines five
type variables:

.. code-block:: python

    class CoatingLayerBase(
        abc.ABC,
        Generic[SubstTypeVar, ParamTypeVar, DrawOptTypeVar, DecoOptTypeVar, DataTypeVar],
    ):
        ...

Its abstract subclass :class:`RectCoatingLayerBase` substitutes
:obj:`SubstTypeVar` with :class:`RectSubstrate`, so that any other
substrate type will be rejected by type checking.

.. code-block:: python

    class RectCoatingLayerBase(
        CoatingLayerBase[
            RectSubstrate, ParamTypeVar, DrawOptTypeVar, DecoOptTypeVar, DataTypeVar
        ]
    ):
        ...

It still has four free type variables, which are substituted by
concrete subclass :class:`RectLayerShape`:

.. code-block:: python

    class RectLayerShape(
        RectCoatingLayerBase[
            RectLayerShapeParam,
            RectLayerShapeDrawOpt,
            RectLayerShapeDecoOpt,
            RectLayerShapeData,
        ]
    ):
        ParamType = RectLayerShapeParam
        DrawOptType = RectLayerShapeDrawOpt
        DecoOptType = RectLayerShapeDecoOpt
        DataType = RectLayerShapeData

Therefore :class:`RectLayerShape` is no longer a generic class.

Additionally, the constant types which replace the type variables
are assigned to special class attributes so that they can be
accessed in runtime. For example, :attr:`ParamType` is used to 
construct default parameter if *parameters* argument is not
provided to the class constructor.

Note that :class:`RectSubstrate` is not set to attribute although
it replaces :obj:`SubstTypeVar`, as it does not need to be
accessed in runtime.

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
