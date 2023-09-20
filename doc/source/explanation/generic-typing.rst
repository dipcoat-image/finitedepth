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

:class:`.CoatingLayerBase` has five type variables; the additional three are ``SubstrateType``, ``DecoOptionsType`` and ``DataType``.

:class:`.ExperimentBase` has two type variables; ``CoatingLayerType`` and ``ParametersType``.

Concrete class
--------------

Classes such as :class:`.Substrate` are concrete classes which fully implement their own base class.
This means that they no longer have type variables.

For example, :class:`.Substrate` has :class:`Parameters <substrate.Parameters>` for ``ParametersType`` and :class:`DrawOptions <substrate.DrawOptions>` for ``DrawOptionsType``,
thus having generic pattern ``SubstrateBase[Parameters, DrawOptions]``.

Partially concrete class
------------------------

:class:`.RectCoatingLayerBase` is an abstract base class which partially implements :class:`CoatingLayerBase` by fixing ``SubstrateType`` to :class:`.RectSubstrate`.
Thus the generic pattern is ``CoatingLayerBase[RectSubstrate, ParametersType, DrawOptionsType, DecoOptionsType, DataType]``.

Its concrete implementation now needs only four more type variables to be fixed.
For example :class:`.RectLayerShape` has generic pattern ``RectCoatingLayerBase[Parameters, DrawOptions, DecoOptions, Data]``.

Implementing concrete classes
=============================

There are three ways to define a new concrete class:

* Implement from abstract class
* Define abstract class and implement from it.
* Inherit the other concrete class.

Which to choose depends on your purpose.

1. If the new class will be a final (no further subclassing), just directly implement concrete class.
2. If you expect your class to be further extended, defining abstract class is the right choice.
3. If you just want to override a few methods, you may inherit from concrete class.

:ref:`user-guide` provides guide to implement concrete class from abstract class.
To know how to define new abstract class, :class:`.RectLayerShape` can serve as a good example.

Inheriting concrete class is generally not recommended because the class design is extremely restricted.
Fixed type values cannot be re-assigned, disallowing any new parameter.
Also, types of method arguments and outputs must be preserved.

.. note::

   Do not use :obj:`typing.TypeAlias` for type variable values in class attribute definition, e.g. ``Parameters``.
   It breaks type inference.
