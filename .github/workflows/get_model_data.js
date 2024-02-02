const fs = require('fs');
const path = require("path");
const yaml = require('js-yaml');

const ignoreDirs = [
    '.datalad', '.github', 'docs', 'images', 'weights', 'trained-models-template'
];

// Corrected the getAllPaths function to properly join paths
const getAllPaths = function(dirPath, arrayOfPaths) {
    const files = fs.readdirSync(dirPath);
    arrayOfPaths = arrayOfPaths || [];

    files.forEach(function(file) {
        if (fs.statSync(path.join(dirPath, file)).isDirectory() && !ignoreDirs.includes(file)) {
            arrayOfPaths = getAllPaths(path.join(dirPath, file), arrayOfPaths);
        } else if (file === 'model_card.yaml') {
            arrayOfPaths.push(path.join(dirPath, file)); // Removed extra "/"
        }
    });

    return arrayOfPaths;
};

function findObj(name, arr) {
    for (const obj of arr) {
        if (obj.name === name) return obj;
    }
    return null;
}

const paths = getAllPaths('.');
const names = [];
const models = {}; // Made 'models' a constant

paths.forEach(function(filePath) {
    const doc = yaml.load(fs.readFileSync(filePath, 'utf8'));

    // Retrieve model details directly
    const modelDetails = doc.Model_details;

    // Destructure the relevant fields from modelDetails
    const { Organization: org, Model_version: version, Model_type: modelType, More_information: modelName } = modelDetails;

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

    // Create models.yml
    const combined_name = [org, modelName, version, modelType].join('_');
    models[combined_name] = doc;

    // Updated modelCardFields to reflect the new structure
    const modelCardFields = [
        'Model_details', 'Intended_use', 'Factors', 'Metrics', 'Evaluation Data', 'Training Data', 'Quantitative Analyses', 'Ethical Considerations', 'Caveats and Recommendations'
    ];

    for (const field of modelCardFields) {
      if (models[combined_name][field] === '') {
        models[combined_name][field] = 'Information not provided.';
      }
    }

    // Create model pages
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
    const filename = `./trained-models-template/docs/_pages/${combined_name}.markdown`;
    fs.writeFile(filename, page, "utf8", err => {
        if (err) console.log(err);
    });
});

// Write to files
const yamlNames = yaml.dump(names);
fs.writeFile("./trained-models-template/docs/_data/names.yml", yamlNames, "utf8", err => {
    if (err) console.log(err);
});

const yamlModels = yaml.dump(models);
fs.writeFile("./trained-models-template/docs/_data/models.yml", yamlModels, "utf8", err => {
    if (err) console.log(err);
});
