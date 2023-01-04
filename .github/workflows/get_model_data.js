const fs = require('fs');
const path = require("path");
const yaml = require('js-yaml');

const ignoreDirs = [
    '.datalad', '.github', 'docs', 'images', 'weights'
];

// find all spec.yml files
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

function findObj(name, arr) {
  for (const obj of arr) {
    if (obj.name === name) return obj;
  }
  return null;
}

const paths = getAllPaths('.');
const names = [];
models = {};

paths.forEach(function(path) {
  const doc = yaml.load(fs.readFileSync(path, 'utf8'));
  
  //create names.yml
  const example = doc.model.example.split(' ');
  let org;
  let modelName;
  let version;
  let modelType;
  for (let i=0; i < example.length; i++) {
    const str = example[i];
    if (str.includes(doc.model.model_name)) {
      const initCombinedName = str.split('/');
      org = initCombinedName[0];
      modelName = initCombinedName[1];
      version = initCombinedName[2];
      modelType = (example[i+1] === '--model_type') ? example[i+2] : 'model';
      break;
    }
  }
  let orgStruct = findObj(org, names);
  if (orgStruct === null) {
    orgStruct = {
      name: org,
      modelNames: []
    };
    names.push(orgStruct);
  }
  let modelNameStruct = findObj(modelName, orgStruct.modelNames);
  if (modelNameStruct === null) {
    modelNameStruct = {
      name: modelName,
      versions: []
    };
    orgStruct.modelNames.push(modelNameStruct);
  }
  let versionStruct = findObj(version, modelNameStruct.versions);
  if (versionStruct === null) {
    versionStruct = {
      name: version,
      modelTypes: []
    };
    modelNameStruct.versions.push(versionStruct);
  }
  versionStruct.modelTypes.push({
    name: modelType
  });

  // create models.yml
  const combined_name = org + '_' + modelName + '_' + version + '_' + modelType;
  models[combined_name] = doc.model;
  modelCardFields = [
    'model_details', 'intended_use', 'factors', 'metrics', 'eval_data', 'training_data', 'quant_analyses', 'ethical_considerations', 'caveats_recs'
  ];
  for (const field of modelCardFields) {
    if (! models[combined_name].hasOwnProperty(field)) {
      models[combined_name][field] = 'Information not provided.';
    }
  }

  // create model pages
  const permalink = `/${org}/${modelName}/${version}/${modelType}/`;
  const page = `---
  layout: model_card
  permalink: ${permalink}
  combined_name: ${combined_name}
  org: ${org}
  modelName: ${modelName}
  version: ${version}
  modelType: ${modelType}
---
  `;
  const filename = `./docs/_pages/${combined_name}.markdown`;
  fs.writeFile(filename, page, "utf8", err => {
    if (err) console.log(err);
  });
});

// write to files
yamlNames = yaml.dump(names);
fs.writeFile("./docs/_data/names.yml", yamlNames, "utf8", err => {
  if (err) console.log(err);
});

yamlModels = yaml.dump(models);
fs.writeFile("./docs/_data/models.yml", yamlModels, "utf8", err => {
    if (err) console.log(err);
});