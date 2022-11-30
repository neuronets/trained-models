const fs = require('fs');
const path = require("path");
const yaml = require('js-yaml');

const ignoreDirs = [
    '.datalad', '.github', 'docs', 'images', 'weights'
];

const getAllPaths = function(dirPath, arrayOfPaths) {
    files = fs.readdirSync(dirPath);

    arrayOfPaths = arrayOfPaths || [];
  
    files.forEach(function(file) {
      if (fs.statSync(dirPath + "/" + file).isDirectory() && !ignoreDirs.includes(file)) {
        arrayOfPaths = getAllPaths(dirPath + "/" + file, arrayOfPaths);
      } else if (file === 'spec.yaml') {
        arrayOfPaths.push(path.join(dirPath, "/", file));
      }
    });
  
    return arrayOfPaths;
}

const paths = getAllPaths('.');
const models = {};
const modelNames = [];
paths.forEach(function(path) {
  const doc = yaml.load(fs.readFileSync(path, 'utf8'));
  if (doc.model.example.includes('--model_type')) {
    const splitExample = doc.model.example.split(' ');
    const index = splitExample.indexOf('--model_type') + 1;
    doc.model.model_type = splitExample[index];
  }
  const model_name = doc.model.model_name;
  const combinedName = (doc.model.model_type) ? `${model_name}_${doc.model.model_type}` : model_name;
  modelNames.push(combinedName);
  models[combinedName] = doc.model;
  const permalink = (doc.model.model_type) ? `/${model_name}/${doc.model.model_type}/` : `/${model_name}/`;
  const page = `---
  layout: model_card
  permalink: ${permalink}
  combined_name: ${combinedName}
  model_name: ${model_name}
---
  `;
  const filename = (doc.model.model_type) ? `./docs/_pages/${model_name}_${doc.model.model_type}.markdown` :
  `./docs/_pages/${model_name}.markdown` 
  fs.writeFile(filename, page, "utf8", err => {
    if (err) console.log(err);
});

});

yamlModelNames = yaml.dump(modelNames);
fs.writeFile("./docs/_data/model_names.yml", yamlModelNames, "utf8", err => {
  if (err) console.log(err);
});

yamlModels = yaml.dump(models);
fs.writeFile("./docs/_data/models.yml", yamlModels, "utf8", err => {
    if (err) console.log(err);
});

