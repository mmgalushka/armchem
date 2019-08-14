#!/bin/bash
#
# Copyright (c) 2017-present, AUROMIND Ltd.

ERROR="\033[1;31m[Error]\033[0m"
DONE="\033[1;32m[Done]\033[0m"
WARN="\033[1;33m[Warn]\033[0m"
INFO="\033[1;34m[Info]\033[0m"

# =============================================================================
# HELPER ACTIONS
# =============================================================================

action_usage(){
    NC=`echo "\033[m"`
    MAIN=`echo "\033[0;39m"` 
    CMD=`echo "\033[1;34m"`
    OPT=`echo "\033[0;34m"`

    echo -e "${MAIN}     _   _   _ ____   ___  __  __ ___ _   _ ____  ${NC}"
    echo -e "${MAIN}    / \ | | | |  _ \ / _ \|  \/  |_ _| \ | |  _ \   ${OPT}ArmCchem${NC}"
    echo -e "${MAIN}   / _ \| | | | |_) | | | | |\/| || ||  \| | | | |  ${OPT}Helper${NC}"
    echo -e "${MAIN}  / ___ \ |_| |  _ <| |_| | |  | || || |\  | |_| |  ${NC}"
    echo -e "${MAIN} /_/   \_\___/|_| \_\\\\\___/|_|  |_|___|_| \_|____/ ${OPT}(c) 2017-present${NC}"
    echo -e "Tool for creating deep learning model for predicting properties of"                                          
    echo -e "chemical compounds based on SMILES."                                          
    echo -e ""                                          
    echo -e "${MAIN}helper ${CMD}command${MAIN} [${OPT}option${MAIN}]${NC}"
    echo -e "\t${CMD}clear${MAIN}\t\t clears environment; ${NC}"
    echo -e "\t${CMD}init${MAIN}\t\t- initializers environment; ${NC}"
    echo -e "\t${CMD}train${OPT} <args>${MAIN}\t- train a model; ${NC}"
    echo -e "\t${CMD}predict${OPT} <args>${MAIN}\t- predict using a model; ${NC}"
}

action_clear(){
    if [ -d .env ];
        then
            rm -r .env
            echo -e "${DONE} Removed virtual environment;"
    fi

    find . -name \*.pyc -delete
    echo -e "${DONE} Removed all '*.pyc' files;"

    find . -name \*.zip -delete
    find . -name \*.tar.gz -delete
    echo -e "${DONE} Removed all '*.(tar.gz|zip)' files;"
}

action_init(){
    action_clear

    virtualenv -q .env --system-site-packages --python=python
    python -V
    which python
    which pip
    echo -e "${DONE} Created virtual environment;"

    source .env/bin/activate
    echo -e "${DONE} Activated virtual environment;"

    python -m pip install -r requirements.txt
    echo -e "${DONE} Installed dependencies;"

    version="$(python -V 2>&1)"
    echo -e "${INFO} Python $version virtual environment installed;"
}

action_train(){
    source .env/bin/activate
    echo -e "${DONE} Activated virtual environment;"

    time python main.py train $@
}

action_predict(){
    source .env/bin/activate
    echo -e "${DONE} Activated virtual environment;"

    time python main.py predict $@
}

action_describe(){
    source .env/bin/activate
    echo -e "${DONE} Activated virtual environment;"

    time python main.py describe $@
}

# =============================================================================
# HELPER COMMANDS SELECTOR
# =============================================================================
case $1 in
    clear)
        action_clear
    ;;
    init)
        action_init
    ;;
    train)
        action_train ${@:2}
    ;;
    predict)
        action_predict ${@:2}
    ;;
    describe)
        action_describe ${@:2}
    ;;
    *)
        action_usage
    ;;
esac  

exit 0