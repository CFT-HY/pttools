.. Please see these for ideas on how to expand this template
   https://github.com/sphinx-doc/sphinx/issues/7912
   https://sphinx-gallery.github.io/stable/configuration.html#auto-documenting-your-api-with-links-to-examples

{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}

   Classes
   -------

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}
       :add-heading:

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}

   Functions
   ---------

   {% for item in functions %}

   .. autofunction:: {{ item }}

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}
       :add-heading:

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}

   Exceptions
   ----------

  {% for item in exceptions %}
   .. autoexception:: {{ item }}

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}

   Attributes
   ----------

  {% for item in attributes %}
   .. autodata:: {{ item }}

   {%- endfor %}
   {% endif %}
   {% endblock %}
