=========================
Class with Generic Typing
=========================

.. currentmodule:: dipcoatimage.finitedepth

This document gives detailed explanation about class design in :mod:`dipcoatimage.finitedepth` and how to extend them.

Generic types
=============

:mod:`dipcoatimage.finitedepth` use abstract generic classes.
This means that base classes have type variables which you should fill in, and abstract methods which you should implement.

Abstract class
--------------

:class:`.SubstrateReferenceBase` and :class:`.SubstrateBase` has two type variables; ``ParametersType`` and ``DrawOptionsType``.
We say that their generic patterns are ``Generic[ParametersType, DrawOptionsType]``.

:class:`.CoatingLayerBase` has five type variables; additional three are ``SubstrateType``, ``DecoOptionsType`` and ``DataType``.

Concrete class
--------------

:class:`.SubstrateReference` and :class:`.Substrate` are concrete classes which fully implement their own base class.
This means that they no longer have type variables.

For example, :class:`.Substrate` has :class:`SubstrateParameters <substrate.SubstrateParameters>` for ``ParametersType`` and :class:`SubstrateDrawOptions <substrate.SubstrateDrawOptions>` for ``DrawOptionsType``,
thus having generic pattern ``SubstrateBase[SubstrateParameters, SubstrateDrawOptions]``.

Simillary, :class:`.LayerArea` implements :class:`.CoatingLayerBase` by having type values for five variable types.

Partial concrete class
----------------------

:class:`.RectCoatingLayerBase` is an abstract base class which partially implements :class:`CoatingLayerBase`.
Its ``SubstrateType`` is fixed to be :class:`.RectSubstrate` and other four variables remain.
Thus the generic pattern is ``CoatingLayerBase[RectSubstrate, ParametersType, DrawOptionsType, DecoOptionsType, DataType]``.

Its implementation now needs only four more types to be concrete.
For example :class:`.RectLayerArea` defines remaining four variables.
Its generic pattern is ``RectCoatingLayerBase[Parameters, DrawOptions, DecoOptions, Data]`` (type names are aliased).


Implementing classes
====================

There are three ways to define a new concrete class:

* Implement from abstract class
* Define abstract class and implement from it.
* Inherit the other concrete class.

Which way to choose depends on your purpose.

1. If you expect your class to be extended, defining abstract class is the right choice.
2. If the new class is one-shot, directly implement concrete class.
3. If you just want to override a few methods, inheriting from concrete class can be good.

:ref:`how-to` provides guide to implement concrete class from abstract class.
To know how to define new abstract class, :class:`.RectLayerArea` can serve as a good example.

.. note::

   Do not use :obj:`typing.TypeAlias` for type variable values in class attribute definition, e.g. ``Paarameters``.
   It breaks type inference.
