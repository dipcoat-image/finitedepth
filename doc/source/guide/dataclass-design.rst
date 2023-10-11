.. _howto-define-dataclass:

How to define dataclass
=======================

.. currentmodule:: finitedepth

Dataclass is profusely used in core classes, and you need to define
your own if you want to :ref:`extend the module <howto-define-class>`.

Dataclasses can be divided into two groups based on their roles:

* Argument dataclass
   These dataclasses wraps the arguments for the core classes.
   They are assigned to :attr:`ParamType`, :attr:`DrawOptType`, and
   :attr:`DecoOptType`.
* Result dataclass
   These dataclasses wraps the analysis result.
   They are assigned to :attr:`DataType` and are returned by
   :meth:`analyze`.

Argument dataclass
------------------

The purpose of argument dataclass is to (de)serialize its data for
configuration file. :obj:`data_converter`, which is a
:class:`cattrs.Converter` is provided for this purpose.

The best practice to make the dataclass convertable is to keep
the types of its fields primitive. In most cases, :obj:`str`,
:obj:`int`, :obj:`float`, :obj:`bool` and :obj:`tuple`
wrapping them are sufficient. If your field picks a value
from limited choices, :class:`enum.Enum` is recommended.

Should your dataclass must store custom type data, you can
register (un)structure hook to :obj:`data_converter`. Refer to
:mod:`cattrs` documentation for more information.

Result dataclass
----------------

Unlike argument dataclass, the result dataclass is not structured
from file. It only needs to be unstructured into result file
such as ``CSV``.

Unstructuring of result dataclass depends on the analysis class
implementation. The basic class, :class:`Analysis`, just uses
:func:`dataclasses.astuple`. It is therefore recommended to design
the result dataclass in flat structure with primitive types.
