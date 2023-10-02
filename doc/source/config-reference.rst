.. _config-reference:

Configuration file reference
============================

.. currentmodule:: finitedepth

A configuration file consists of nested mapping entries, each top level item
representing individual analysis suite. The key is the title of the suite
and the value is its data, which corresponds to the parameters of
:class:`ConfigBase`.

For example, the following YAML-format configuration file includes two
analysis suites:

.. code-block:: YAML

    Title1:
        ...

    Title2:
        ...

When :ref:`command-line analysis <basic-example>` is invoked, each analysis
suite is automatically structured to :class:`Config` instance using
:obj:`data_converter`. Alternatively, user can directly construct the instance
in runtime and call its :meth:`~serialize.ConfigBase.analyze` method.

.. note::

    You can define your own configuration format which does not follow the
    structure described here. Refer to :ref:`howto-extend-config` page.

This document describes the members of analysis suite data.
Each subsection describes the member of upper-level item.
Refer to :ref:`tutorial` page for working examples.

ref_path
--------

Type: String.

Path to a file which stores reference image.

Scope of the file format depends on the :class:`ConfigBase` implementation.

The reference instance is constructed from this reference image and
data from ``reference`` field.

coat_path
---------

Type: String.

Path to a file which stores taret image(s).

Scope of the file format depends on the :class:`ConfigBase` implementation.

The coating layer instances are constructed from these coating layer images
and data from ``coatinglayer`` field, using experiment instance from
``experiment`` field as factory.

reference
---------

Type: Map, optional.

Type and arguments to construct :class:`ReferenceBase` implementation.
Defaults to construct :class:`Reference` instance.

The reference instance is constructed from this data and reference image
from ``ref_path`` field.

.. _config-ref-type:

type
^^^^

Type: Map, optional.

Data to import a reference type. Defaults to import :class:`Reference`.

This map consists of variable name and module name to import an object,
which must be a concrete implementation of :class:`ReferenceBase`.

name
""""

Type: String, optional.

Name of the reference class. Defaults to ``"Reference"``.

module
""""""

Type: String, optional.

Module from which the reference class is imported. Defaults to
``"dipcoatimage.finitedepth"``.

templateROI
^^^^^^^^^^^

Type: List of four Integers, optional

ROI in the reference image to define template region. Defaults to full ROI
(entire image).

substrateROI
^^^^^^^^^^^^

Type: List of four Integers, optional

ROI in the reference image to define substrate region. Defaults to full ROI
(entire image).

parameters
^^^^^^^^^^

Type: Map, optional.

Parameters of the reference instance. Defaults to an empty map.

Parameters of a reference instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`Parameters <reference.ReferenceBase.Parameters>`.
This map is structured by :obj:`data_converter` to construct the parameter
dataclass instance.

As each implementation of :class:`ReferenceBase` define its own
:attr:`Parameters <reference.ReferenceBase.Parameters>`, the members of this
map depend on :ref:`config-ref-type` field.

draw_options
^^^^^^^^^^^^

Type: Map, optional.

Drawing options of the reference instance. Defaults to an empty map.

Drawing options of a reference instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`DrawOptions <reference.ReferenceBase.DrawOptions>`.
This map is structured by :obj:`data_converter` to construct the drawing
option dataclass instance.

As each implementation of :class:`ReferenceBase` define its own
:attr:`DrawOptions <reference.ReferenceBase.DrawOptions>`, the members of
this map depend on :ref:`config-ref-type` field.

substrate
---------

Type: Map, optional.

Type and arguments to construct :class:`SubstrateBase` implementation.
Defaults to construct :class:`Substrate` instance.

The substrate instance is constructed from this data and reference instance
from ``ref_path`` and ``reference`` fields.

.. _config-subst-type:

type
^^^^

Type: Map, optional.

Data to import a substrate type. Defaults to import :class:`Substrate`.

This map consists of variable name and module name to import an object,
which must be a concrete implementation of :class:`SubstrateBase`.

name
""""

Type: String, optional.

Name of the substrate class. Defaults to ``"Substrate"``.

module
""""""

Type: String, optional.

Module from which the substrate class is imported. Defaults to
``"dipcoatimage.finitedepth"``.

parameters
^^^^^^^^^^

Type: Map, optional.

Parameters of the substrate instance. Defaults to an empty map.

Parameters of a substrate instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`Parameters <substrate.SubstrateBase.Parameters>`.
This map is structured by :obj:`data_converter` to construct the parameter
dataclass instance.

As each implementation of :class:`SubstrateBase` define its own
:attr:`Parameters <substrate.SubstrateBase.Parameters>`, the members of
this map depend on :ref:`config-subst-type` field.

draw_options
^^^^^^^^^^^^

Type: Map, optional.

Drawing options of the substrate instance. Defaults to an empty map.

Drawing options of a substrate instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`DrawOptions <substrate.SubstrateBase.DrawOptions>`.
This map is structured by :obj:`data_converter` to construct the drawing
option dataclass instance.

As each implementation of :class:`SubstrateBase` define its own
:attr:`DrawOptions <substrate.SubstrateBase.DrawOptions>`, the members of
this map depend on :ref:`config-subst-type` field.

coatinglayer
------------

type
^^^^

name
""""

module
""""""

parameters
^^^^^^^^^^

draw_options
^^^^^^^^^^^^

deco_options
^^^^^^^^^^^^

experiment
----------

type
^^^^

name
""""

module
""""""

parameters
^^^^^^^^^^

analysis
--------

type
^^^^

name
""""

module
""""""

parameters
^^^^^^^^^^

fps
^^^
