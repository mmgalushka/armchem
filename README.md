xcode-select --install

# Installation

To install the Python virtual environment and all required packages, go to `auromind` folder and execute the following command:

```bash
$ ./helper.sh init
```

To clear the Python virtual environment with all artifacts created during experimentation such as logs, model, data etc. execute the following command:

```bash
$ ./helper.sh clear
```

**Note:** Make sure to copy your working artifacts to a directory outside the project since they will be removed as a result of this operation. 