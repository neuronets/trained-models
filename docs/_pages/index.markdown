---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
permalink: /
---
<div class="models-body">
    <div class="models-list">
        {% for org in site.data.names %}
            {% include org.html header=org %}
            {% for modelName in site.data.names[org] %}
                {% include modelName.html header=modelName %}
                {% for version in site.data.names[org][modelName] %}
                    {% include version.html header=version %}
                    {% for modelType in site.data.names[org][modelName][version] %}
                        {% assign model = site.data.models[org][modelName][version][modelType] %}
                        {% assign header = model.model_name %}
                        {% assign subheader = model.model_type %}
                        {% include model.html header=header subheader=subheader %}
                    {% endfor %}
                {% endfor %}
            {% endfor %}
        {% endfor %} 
    </div>
</div>