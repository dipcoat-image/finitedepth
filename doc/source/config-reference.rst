.. _config-reference:

Configuration file reference
============================

.. currentmodule:: finitedepth

A configuration file consists of nested mapping entries. Each top level item
represent individual analysis suite; the key is the title of the suite
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

    It is possible for you to define your own configuration. Refer to
    :ref:`howto-extend-config` page.

In the rest of this document, members of analysis suite data are described.
Each subsection describes the member of upper-level map. Refer to the
:ref:`tutorial` page for examples.

ref_path
--------

Type: String.

Path to the file to get reference image.

Scope of the file format depends on the :class:`ConfigBase` implementation.

coat_path
---------

Type: String.

Path to the file to get taret image(s).

Scope of the file format depends on the :class:`ConfigBase` implementation.

reference
---------

Type: Map, optional.

Parameters for reference instance.

With the reference image constructed by ``ref_path``, the data in this map
construct instance of :class:`ReferenceBase` implementation.

.. _config-ref-type:

type
^^^^

Type: Map, optional.

Specify concrete class of :class:`ReferenceBase`.

This map consists of variable name and module name to import the class.
Default data imports :class:`Reference`.

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

Data for :attr:`ReferenceBase.Parameters <reference.ReferenceBase.Parameters>`.

This map is structured by :obj:`data_converter` to construct the instance of
:attr:`~reference.ReferenceBase.Parameters` attribute of :class:`ReferenceBase`
implementation, specified by the :ref:`config-ref-type` field.

The members of this map depends on the reference type.

draw_options
^^^^^^^^^^^^

Type: Map, optional.

Data for :attr:`ReferenceBase.DrawOptions <reference.ReferenceBase.DrawOptions>`.

This map is structured by :obj:`data_converter` to construct the instance of
:attr:`~reference.ReferenceBase.DrawOptions` attribute of :class:`ReferenceBase`
implementation, specified by the :ref:`config-ref-type` field.

The members of this map depends on the reference type.

substrate
---------

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
