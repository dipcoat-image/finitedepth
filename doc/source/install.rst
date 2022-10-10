============
Installation
============

This document explains how to install DipCoatImage-FiniteDepth.

Making virtual environment
==========================

It is recommended to make a dedicated virtual environment.
The easiest way is to use `Anaconda <https://www.anaconda.com/>`_:

.. code-block:: bash

   $ conda create -n dipcoat-image pip
   $ conda activate dipcoat-image

You are now in a new environment "dipcoat-image", with only `pip <https://pip.pypa.io/en/stable/>`_ package installed.
Ready to go!

Downloading the source (Optional)
=================================

You can download full source code of DipCoatImage-FiniteDepth project without installing it by git.

.. code-block:: bash

   $ git clone git@github.com:dipcoat-image/dipcoat-image/finitedepth.git

Note that you can download the source with ``pip`` command, but it will install the package at the same time.
It will be explaned in the next section.

Setting environment variable (Optional)
=======================================

DipCoatImage-FiniteDepth is dependent to PySide6 by default.
This can cause trouble if you are running in non-GUI environment (i.e. in server), or with other packages dependent to Qt.
For example, non-headless OpenCV-Python modifies Qt library, making PySide6 unavailable.

To install non-GUI (headless) version, set the environment variable ``DIPCOATIMAGE_HEADLESS`` to ``1`` before installing the package.

For example, in Linux:

.. code-block:: bash

   $ export DIPCOATIMAGE_HEADLESS=1

In Windows CMD:

.. code-block:: console

   > set DIPCOATIMAGE_HEADLESS=1

Installing
==========

The package can be installed by

.. code-block:: bash

   $ pip install [-e] <url/path>[dependency options]

If you just want quick installation for user without source, the following command will be enough.

.. code-block:: bash

   $ pip install git+ssh://git@github.com/dipcoat-image/finitedepth.git

This will install ``dipcoatimage-finitedepth`` package which consists of :mod:`dipcoatimage.finitedepth` and :mod:`dipcoatimage.finitedepth_gui` in your environment.

If you have set the environment variable to install the headless version, ``dipcoatimage-finitedepth-headless`` package will be installed instead.
This package contains :mod:`dipcoatimage.finitedepth` only (no :mod:`dipcoatimage.finitedepth_gui`).

.. rubric:: Install options

There are two types of install options for developers.

* Install with editable option (``-e``)
* Install with dependency specification (``[...]``)

Editable option installs the package as link to the original location.
Change to the source directly reflects to your environment.

Dependency specification installs additional modules which are required to access extra features of the package.
You may add them in brackets right after the package argument.

Available specifications are:

* ``test``: installs modules to run tests
* ``test-ci``: installs modules to run tests in headless environment.
* ``doc``: installs modules to build documentations
* ``full``: installs every additional dependency

With commas without trailing whitespaces, i.e. ``[A,B]``, you can pass multiple specifications.

Installing from repository
--------------------------

By passing the vcs url, ``pip`` command automatically clones the source code and installs the package.

.. code-block:: bash

   $ pip install git+ssh://git@github.com/dipcoat-image/finitedepth.git

If you want to pass install options, you need to specify the package name by ``#egg=``.
For example, the following code installs the package with every additional dependency.

.. code-block:: bash

   $ pip install git+ssh://git@github.com/dipcoat-image/finitedepth.git#egg=dipcoatimage-finitedepth[full]

.. note::

   If you pass ``-e`` option, full source code of the project will be downloaded under ``src/`` directory in your current location.

Installing from source
----------------------

If you have already downloaded the source, you can install it by passing its path to ``pip install``.
For example, in the path where ``setup.py`` is located the following command installs the package in editable mode, with full dependencies.

.. code-block:: bash

   $ pip install -e .[full]
