## Instruction of adding a model to trained_model repository

1. we suggest to create a separate environment to submit your model.
- install and configure [git](https://github.com/git-guides/install-git), [git-annex](https://git-annex.branchable.com/install/) and also you need a github account.
- Install and configure [datalad](https://handbook.datalad.org/en/latest/intro/installation.html)
- install [datalad-osf](https://handbook.datalad.org/en/latest/intro/installation.html) and get your credentials.

2. Fork and then clone the trained_model repository from your Fork. You can use either git or datalad.

	Using git:```git clone git@github.com:<your_github>/trained-models.git```

	or

	Using datalad:```datalad clone git@github.com:<your_github>/trained-models.git```

3. Add the osf-storage remote as a publish dependency to the github.

	```datalad siblings configure -s origin --publish-depends osf-storage```
	
4. **NOTE: do not create a new branch. Work on master branch.** The structure of the repository is ```<org>/<model_name>/<version>```. Create these nested directories with proper names and version of your model. Organization name can refer to your affiliation. You need below files for adding your model,
	- **saved model file or weights:** add the model file in a subdirectory called ```weights```
	- **sep.yaml:** this file contains an information about your model, command to run, training dataset and etc. check the template [here](). 
	- **prediction.py:** script for getting inference from the model
	- **train.py:** script for training of the model 
	- **requirement.txt:** list of model dependencies for inference and train
	- **Dockerfile:** docker file to build a container

You may submit your model only for inference, in that case you may not have the ```train.py``` file.

5. Save the dataset with a commit message from the root of the repository.
	
	```datalad save -m “Added model X”```
			
6. Push the changes to the github.
	
	```datalad push --to origin```
	
7. Send the pull request to the [neuronet/trained_model](git@github.com:neuronets/trained-models.git) repository.
	
	**NOTE: you should send two pull request, one for master branch and one for git-annex branch.**

