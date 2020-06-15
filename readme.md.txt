1)This repository contains Two workflows.One workflow is to triggered for every github action to valiadate whether the model 
runs without error and also checks for any flake 8 erros.
2) Another workflow is triggered  for tagged release and creates a model for entire data set and uploads resultant model to s3.
3.I created an amazon account and created in a s3 container to store the model.
4) aws access key id,aws access secret key and awd bucket information are stored in the secrets and can be accessed by
using secerets.AWS_ACCESS_KEY_ID, secrets.AWS_SECRET_ACCESS_KEY,secrets.AWS_BUCKET_NAME from workflows
**************
On any Pull request. A github action is triggered and below steps are executed.
1.First existing project is checked out in to the virtual machine and python 3.8 is installed. after that pip, flake 8 , flake 8 doc strings
 are installed
2.this actions check for any default flake 8 errors and missing doc strings from all the python files in the repository.
3 once this job is done, then as a part of existing github actions new job will start which will install python 3.8 first and 
all the python dependencies from requirements.txt file , after that we will run main.py file with an argument "test"
4.Minimal dataset for any PR is ran to make sure the program runs without error.So the command
 line argument "Test" is used to run the python file for minimal data.
5.If the sys.argv[1] is 'Test' then a model is created with the minimal data set of 20 rows.
this github action will complete successfully only if all above steps are run successfully, other wise it will fail.
******************
on any pull request which is tagged as release a new github action is triggered with below steps
1.First existing project is checked out in to the virtual machine and python 3.8 is installed. It also checksout 3syncaction to upload model to s3
2. once this job is done, then as a part of existing github actions new job will start which will install python 3.8 first and 
all the python dependencies from requirements.txt file , after that we will run main.py file 
3. Python program is run with all data in the dataset and resulting model is saved to s3 using "jakejarvis/s3-sync-action@master"

