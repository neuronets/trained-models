## Instruction of adding a model to trained_model repository

1. Create the environment : we suggest to create a separate environment to submit your model
- You need to install and configure [git](https://github.com/git-guides/install-git) and [git-annex](https://git-annex.branchable.com/install/) and also you need a github account
- Install and configure [datalad](https://handbook.datalad.org/en/latest/intro/installation.html)
- install [datalad-osf](https://handbook.datalad.org/en/latest/intro/installation.html) and get your credentials.
First fork and then clone the Trained_model repository from your Fork. You can use either git or datalad.
Using git:	git clone git@github.com:<your_github>/trained-models.git
or
Using datalad: datalad clone git@github.com:<your_github>/trained-models.git
Add the osf remote as the publish dependency 
datalad siblings configure -s origin --publish-depends osf-storage
Note : do not create a new branch. Work on master branch.	
The structure of the repository is <org>/<model_name>/<version>/<model_file>. Copy or move the trained model in the proper location based on the above structure.
Save the dataset. From the root of the repository,
Datalad save -m “Added model X”
			
Push the changes to the github,
Datalad push –-to origin
Send the pull request to the neuronet repository.
Note: you should send two PR one for master branch and one for git-annex branch

