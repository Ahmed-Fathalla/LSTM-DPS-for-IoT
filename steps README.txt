First of all, the required liberaries listed in "requirements.txt" are needed to be installed.

To run a new experiment:
	- Intel-DS: make cmd execution:   python "a. run Intel-DS.py"
	- Benzene-DS: make cmd execution: python "b. run Benzene-DS.py"
	
To check experiment info of an experiment reported in the paper:
	- make cmd execution: python "c. load_custom_experiment.py", edit the experiment string in the function call inside the script file

To re-run an experiment and get the reproducable results as reported in the paper (using the same model we got during our experiment):
	- make cmd execution: python "d. re_run exp.py", edit the experiment string in the function call inside the script file