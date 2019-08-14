As a result of the conducted research, AUROMIND Ltd. released the project **ArmChem** under MIT License. **ArmChem** is a tool for training [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) models for predicting properties of chemical compounds defined by [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system).



# Install Environment

To install the Python virtual environment and all required packages, go to `armchem` folder and execute the following command:

```bash
$ ./helper.sh init
```

If execution of this command is successful


**Note:** 
For Mac OS it might be required to run ```xcode-select --install``` before running initialization command.



# Purge Environment

To clear the Python virtual environment with all artifacts created during experimentation such as logs, model, data etc. execute the following command:

```bash
$ ./helper.sh clear
```

**Note:** 
Make sure to copy your working artifacts (such as `datasets`, `models`, `experiments`,`logs` etc.) to a location outside the project since they will be removed as a result of both operations.