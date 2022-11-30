---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---
<link rel="stylesheet" type="text/css" href="/assets/styles/main.css">
{% include nav.html %}
<div class="models-body">
    <h2 class="models-title">Models</h2>
    <div class="models-list">
        {% for model in site.data.models %}
        {% assign header = model.model_name %}
        {% assign subheader = model.model_type %}
        {% include model.html header=header subheader=subheader %}
        {% endfor %}
    </div>
</div>