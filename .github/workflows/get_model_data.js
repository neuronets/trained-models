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
/* const models = [];
paths.forEach(function(path) {
    const doc = yaml.load(fs.readFileSync(path, 'utf8'));
    if (doc.model.example.includes('--model_type')) {
      const splitExample = doc.model.example.split(' ');
      const index = splitExample.indexOf('--model_type') + 1;
      doc.model.model_type = splitExample[index];
    }
    models.push(doc.model);
}); */
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
  modelNames.push(model_name);
  models[model_name] = doc.model;
});

yamlModelNames = yaml.dump(modelNames);
fs.writeFile("./docs/_data/model_names.yml", yamlModelNames, "utf8", err => {
  if (err) console.log(err);
});

yamlModels = yaml.dump(models);
fs.writeFile("./docs/_data/models.yml", yamlModels, "utf8", err => {
    if (err) console.log(err);
});