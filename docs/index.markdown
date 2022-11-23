---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
<ul>
    {% for model in site.data.models %}
    <li class="model">{{ model.model_name }}</li>
    {% endfor %}
</ul>