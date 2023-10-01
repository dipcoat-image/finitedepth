.. _api:

API Reference
=============

This page contains auto-generated API reference documentation [#f1]_.

Modules are imported under ``dipcoatimage`` namespace.
For example, :mod:`finitedepth` can be accessed by:

.. code-block:: Python

   import dipcoatimage.finitedepth

.. toctree::
   :titlesonly:

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

.. [#f1] Created with `sphinx-autoapi <https://github.com/readthedocs/sphinx-autoapi>`_
