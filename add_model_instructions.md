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
	
4. The structure of the repository is ```<org>/<model_name>/<version>/<model_file>```. Copy or move the trained model in the proper location based on this structure.
	
	**NOTE: do not create a new branch. Work on master branch.**
	
5. Save the dataset with a commit message from the root of the repository.
	
	```datalad save -m “Added model X”```
			
6. Push the changes to the github.
	
	```datalad push --to origin```
	
7. Send the pull request to the [neuronet/trained_model](git@github.com:neuronets/trained-models.git) repository.
	
	**NOTE: you should send two pull request, one for master branch and one for git-annex branch.**

