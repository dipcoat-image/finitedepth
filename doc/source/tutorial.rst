.. _tutorial:

Tutorial
========

.. note::

    To run this tutorial, environment variable ``$FINITEDEPTH_SAMPLES`` must be set:

    .. tabs::

        .. code-tab:: bash

            export FINITEDEPTH_SAMPLES=$(finitedepth samples)

        .. code-tab:: bat cmd

            FOR /F %G IN ('finitedepth samples') DO SET FINITEDEPTH_SAMPLES=%G

        .. code-tab:: powershell

            $env:FINITEDEPTH_SAMPLES=$(finitedepth samples)

    Check if the variable is properly set.
    The output of ``finitedepth samples`` command should be same as the result of:

    .. tabs::

        .. code-tab:: bash

            echo $FINITEDEPTH_SAMPLES

        .. code-tab:: bat cmd

            echo %FINITEDEPTH_SAMPLES%

        .. code-tab:: powershell

            echo $env:FINITEDEPTH_SAMPLES

Download :download:`example.yml` file in your local directory.
The contents of this configuration file are:

.. literalinclude:: example.yml
    :language: yaml

Running the following command will analyze the files::

    finitedepth analyze example.yml

Result (``example1.png``):

.. image:: _static/example1.png
   :width: 400
   :alt: Analysis result.
