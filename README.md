# Amazon Review sentiment analysis

Project created on learning purposes to get familiar with sentiment analysis and working with
dvc and Docker. To build and open project from Docker container run:
> bash cmd_docker.sh

Then make sure the dvc is initialized and run:
> dvc repro

to reproduce all steps mentioned in dvc.yaml. You have several evaluation params that could be changed:
1. strategy: uniform / other
2. data_selector: text / all / other
3. classifier: rf / dummy / svm

By default they are set to:
1. strategy: uniform 
2. data_selector: text 
3. classifier: rf 




