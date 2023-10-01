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

    It is possible for user to define their own configuration.
    Refer to :ref:`howto-extend-config` page.

In the rest of this document, members of analysis suite data are described.

ref_path
--------

Type: ``String``.

coat_path
---------

Type: ``String``.

reference
---------

Type: ``Map``, optional.

type
^^^^

name
""""

module
""""""

templateROI
^^^^^^^^^^^

substrateROI
^^^^^^^^^^^^

parameters
^^^^^^^^^^

draw_options
^^^^^^^^^^^^

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
