---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
permalink: /
---
<div class="models-body">
    <h2 class="models-title">Models</h2>
    <div class="models-list">
        {% for model_name in site.data.model_names %}
        {% assign model = site.data.models[model_name] %}
        {% assign header = model.model_name %}
        {% assign subheader = model.model_type %}
        {% include model.html header=header subheader=subheader %}
        {% endfor %} 
    </div>
</div>