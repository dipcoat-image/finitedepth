.. _howto-runtime:

How to analyze in runtime
=========================

.. currentmodule:: finitedepth

When :ref:`command-line analysis <basic-example>` is invoked, instances of
the following classes are constructed for the analysis:

#. :ref:`Reference class <howto-reference>`
#. :ref:`Substrate class <howto-substrate>`
#. :ref:`Coating layer class <howto-coatinglayer>`
#. :ref:`Experiment class <howto-experiment>`
#. :ref:`Analysis class <howto-analysis>`

In this page, detailed instructions to construct and use these objects in
runtime are provided.

.. _howto-reference:

Reference instance
------------------

Reference instance wraps *reference image*, which is an image of bare
substrate.

.. _howto-substrate:

Substrate instance
------------------

Substrate instance handles substrate region in bare substrate image,
which is defined by reference instance.

.. _howto-coatinglayer:

Coating layer instance
----------------------

Coating layer instance handles *target image*, which is an image of
coated substrate.

.. _howto-experiment:

Experiment instance
-------------------

Experiment instance is a factory object to construct consecutive
coating layer instances.

.. _howto-analysis:

Analysis instance
-----------------

Analysis instance provides coroutine to store the analysis result
as files.

Analyze!
--------

Analysis is performed by constructing the coating layer instances
(preferably using experiment instance) and sending them to the
analysis coroutine,
