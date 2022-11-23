---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
{% for model in site.data.models %}
<h1>{{ model.model_name }}</h1>
<ul>
    <li>{{ model.description }}</li>
    <li>Structure: {{ model.structure }}</li>
    <li>Training Mode: {{ model.training_mode }}</li>
</ul>
{% endfor %}