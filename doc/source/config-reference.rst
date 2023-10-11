.. _config-reference:

Configuration file reference
============================

.. currentmodule:: finitedepth

A configuration file consists of nested maps, each top-level entry
representing an individual set of experiment and analysis. The key of the
top-level map is its title and the value is its data, which corresponds to
the arguments of :class:`ConfigBase`.

For example, the following YAML-format configuration file includes two sets
of data:

.. code-block:: YAML

    Title1:
        ...

    Title2:
        ...

When :ref:`command-line analysis <basic-example>` is invoked, each entry is
automatically structured to :class:`Config` instance using
:obj:`data_converter`. Alternatively, user can directly construct the
configuration instance in runtime and call its
:meth:`~serialize.ConfigBase.analyze` method.

This document describes the data structure which applies to every concrete
subclass of :class:`ConfigBase`. Refer to :ref:`tutorial` page for working
examples.

.. note::

    You can implement your own :class:`ConfigBase` subclass other than
    :class:`Config`. Refer to :ref:`howto-define-class` page.

ref_path
--------

Type: String.

Path to a file which stores reference image.

Scope of the file format depends on :class:`ConfigBase` implementation.

The reference instance is constructed from this reference image and
``reference`` field.

coat_path
---------

Type: String.

Path to a file which stores taret image(s).

Scope of the file format depends on :class:`ConfigBase` implementation.

The coating layer instances are constructed from these target images and
``coatinglayer`` field, using experiment instance from ``experiment`` field
as a factory.

reference
---------

Type: Map, optional.

Type and arguments to construct reference instance.
Defaults to construct :class:`Reference` instance.

The reference instance is constructed from this field and reference image
from ``ref_path`` field.

.. _config-ref-type:

type
^^^^

Type: Map, optional.

Data to import a reference type. Defaults to import :class:`Reference`.

This map consists of variable name and module name to import an object,
which must be a concrete subclass of :class:`ReferenceBase`.

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

Type and arguments to construct substrate instance.
Defaults to construct :class:`Substrate` instance.

The substrate instance is constructed from this field and reference
instance from ``ref_path`` and ``reference`` fields.

.. _config-subst-type:

type
^^^^

Type: Map, optional.

Data to import a substrate type. Defaults to import :class:`Substrate`.

This map consists of variable name and module name to import an object,
which must be a concrete subclass of :class:`SubstrateBase`.

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

Type: Map, optional.

Type and arguments to construct coating layer instance.
Defaults to construct :class:`CoatingLayer` instance.

The coating layer instances are constructed from this field and the target
images from ``coat_path`` field, using experiment instance from ``experiment``
field as a factory.

.. _config-coat-type:

type
^^^^

Type: Map, optional.

Data to import a coating layer type. Defaults to import
:class:`CoatingLayer`.

This map consists of variable name and module name to import an object,
which must be a concrete subclass of :class:`CoatingLayerBase`.

name
""""

Type: String, optional.

Name of the coating layer class. Defaults to ``"CoatingLayer"``.

module
""""""

Type: String, optional.

Module from which the coating layer class is imported. Defaults to
``"dipcoatimage.finitedepth"``.

parameters
^^^^^^^^^^

Type: Map, optional.

Parameters of the coating layer instance. Defaults to an empty map.

Parameters of a coating layer instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`Parameters <coatinglayer.CoatingLayerBase.Parameters>`.
This map is structured by :obj:`data_converter` to construct the parameter
dataclass instance.

As each implementation of :class:`CoatingLayerBase` define its own
:attr:`Parameters <coatinglayer.CoatingLayerBase.Parameters>`, the members
of this map depend on :ref:`config-coat-type` field.

draw_options
^^^^^^^^^^^^

Type: Map, optional.

Drawing options of the coating layer instance. Defaults to an empty map.

Drawing options of a coating layer instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`DrawOptions <coatinglayer.CoatingLayerBase.DrawOptions>`.
This map is structured by :obj:`data_converter` to construct the drawing
option dataclass instance.

As each implementation of :class:`CoatingLayerBase` define its own
:attr:`DrawOptions <coatinglayer.CoatingLayerBase.DrawOptions>`, the members
of this map depend on :ref:`config-coat-type` field.

deco_options
^^^^^^^^^^^^

Type: Map, optional.

Decorating options of the coating layer instance. Defaults to an empty map.

Decorating options of a coating layer instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`DecoOptions <coatinglayer.CoatingLayerBase.DecoOptions>`.
This map is structured by :obj:`data_converter` to construct the decorating
option dataclass instance.

As each implementation of :class:`CoatingLayerBase` define its own
:attr:`DecoOptions <coatinglayer.CoatingLayerBase.DecoOptions>`, the members
of this map depend on :ref:`config-coat-type` field.

experiment
----------

Type: Map, optional.

Type and arguments to construct experiment instance.
Defaults to construct :class:`Experiment` instance.

The experiment instance is a factory object for the coating layer instances,
and is constructed solely from this field.

.. _config-expt-type:

type
^^^^

Type: Map, optional.

Data to import an experiment type. Defaults to import :class:`Experiment`.

This map consists of variable name and module name to import an object,
which must be a concrete subclass of :class:`ExperimentBase`.

name
""""

Type: String, optional.

Name of the experiment class. Defaults to ``"Experiment"``.

module
""""""

Type: String, optional.

Module from which the experiment class is imported. Defaults to
``"dipcoatimage.finitedepth"``.

parameters
^^^^^^^^^^

Type: Map, optional.

Parameters of the experiment instance. Defaults to an empty map.

Parameters of a experiment instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`Parameters <experiment.ExperimentBase.Parameters>`.
This map is structured by :obj:`data_converter` to construct the parameter
dataclass instance.

As each implementation of :class:`ExperimentBase` define its own
:attr:`Parameters <experiment.ExperimentBase.Parameters>`, the members
of this map depend on :ref:`config-expt-type` field.

analysis
--------

Type: Map, optional.

Type and arguments to construct analysis instance.
Defaults to construct :class:`Analysis` instance.

The analysis instance saves the analysis results as file,
and is constructed solely from this field.

.. _config-analysis-type:

type
^^^^

Type: Map, optional.

Data to import an analysis type. Defaults to import :class:`Analysis`.

This map consists of variable name and module name to import an object,
which must be a concrete subclass of :class:`AnalysisBase`.

name
""""

Type: String, optional.

Name of the analysis class. Defaults to ``"Analysis"``.

module
""""""

Type: String, optional.

Module from which the analysis class is imported. Defaults to
``"dipcoatimage.finitedepth"``.

parameters
^^^^^^^^^^

Type: Map, optional.

Parameters of the analysis instance. Defaults to an empty map.

Parameters of a analysis instance are encapsulated and passed as a
:func:`Dataclass <dataclasses.dataclass>` instance, whose type is defined
as class attribute
:attr:`Parameters <analysis.AnalysisBase.Parameters>`.
This map is structured by :obj:`data_converter` to construct the parameter
dataclass instance.

As each implementation of :class:`AnalysisBase` define its own
:attr:`Parameters <analysis.AnalysisBase.Parameters>`, the members
of this map depend on :ref:`config-analysis-type` field.
