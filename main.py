# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

# ===[ IMPORTS ] ======================================
import re
import os
import yaml
import sys
import argparse
import time
import tempfile

from armchem import (
    smiles,
    numeric,
    category
)

from armchem import (
    read_records,
    split_dataset,
    load_network,
    create_autoencoder,
    create_regressor,
    create_classifier,
    create_dataset,
    create_estimator,
    run_single_validation,
    save_object
)

from armchem import (
    load_object,
    print_summary
)
# =====================================================

# ===[ LOGGER ] =======================================
import logging
import logging.config

if os.path.exists('logging.yaml'):
    with open('logging.yaml', 'rt') as f:
        logging.config.dictConfig(
            yaml.safe_load(f.read())
        )
else:
    logging.basicConfig(level=logging.INFO)
# =====================================================


# =====================================================
# Command Methods
# =====================================================

def runCmdTrain(argv):
    """Trains a model."""
    parser = argparse.ArgumentParser(
        prog='train', description='Train a model.'
    )

    train_entity = parser.add_subparsers(
        title='entity',
        dest='entity'
    )

    # -----------------------------------------------------------
    # Defines command-line options for train an autoencoder.
    # -----------------------------------------------------------
    autoencoder_parser = train_entity.add_parser('autoencoder')

    autoencoder_parser.add_argument(
        'dataset',
        metavar='FILE',
        help='input dataset file;'
    )

    autoencoder_parser.add_argument(
        'model',
        metavar='DIR',
        nargs='?',
        help='output model directory;',
        default='models'
    )

    autoencoder_parser.add_argument(
        'experiment',
        metavar='DIR',
        nargs='?',
        help='output experiment directory;',
        default='experiments'
    )

    autoencoder_parser.add_argument(
        '-e', '--epochs',
        metavar='NUM',
        type=int,
        required=False,
        help='set maximum number of training epochs;',
        default=100
    )
    autoencoder_parser.add_argument(
        '-b', '--batch-size',
        metavar='NUM',
        type=int,
        required=False,
        help='set batch size;',
        default=32
    )
    autoencoder_parser.add_argument(
        '--max-smiles-length',
        metavar='NUM',
        type=int,
        required=False,
        help='set maximum SMILES length;',
        default=150
    )
    autoencoder_parser.add_argument(
        '--in-memory',
        action='store_const',
        const=True,
        help='set computation in memory;',
        default=False
    )

    autoencoder_parser.add_argument(
        '-v', '--verbose',
        action='store_const',
        const=True,
        help='set verbose mode;',
        default=False
    )

    # -----------------------------------------------------------
    # Defines command-line options for train a classifier.
    # -----------------------------------------------------------
    classifier_parser = train_entity.add_parser('classifier')

    classifier_parser.add_argument(
        'dataset',
        metavar='FILE',
        help='input dataset file;'
    )

    classifier_parser.add_argument(
        'autoencoder',
        metavar='FILE',
        help='autoencoder config file;'
    )

    classifier_parser.add_argument(
        'model',
        metavar='DIR',
        nargs='?',
        help='output model directory;',
        default='models'
    )

    classifier_parser.add_argument(
        'experiment',
        metavar='DIR',
        nargs='?',
        help='output experiment directory;',
        default='experiments'
    )

    classifier_parser.add_argument(
        '-e', '--epochs',
        metavar='NUM',
        type=int,
        required=False,
        help='set maximum number of training epochs;',
        default=100
    )
    classifier_parser.add_argument(
        '-b', '--batch-size',
        metavar='NUM',
        type=int,
        required=False,
        help='set batch size;',
        default=32
    )
    classifier_parser.add_argument(
        '--max-smiles-length',
        metavar='NUM',
        type=int,
        required=False,
        help='set maximum SMILES length;',
        default=150
    )
    classifier_parser.add_argument(
        '--in-memory',
        action='store_const',
        const=True,
        help='set computation in memory;',
        default=False
    )

    classifier_parser.add_argument(
        '-v', '--verbose',
        action='store_const',
        const=True,
        help='set verbose mode;',
        default=False
    )

    # -----------------------------------------------------------
    # Defines command-line options for train a regressor.
    # -----------------------------------------------------------
    regressor_parser = train_entity.add_parser('regressor')

    regressor_parser.add_argument(
        'dataset',
        metavar='FILE',
        help='input dataset file;'
    )

    regressor_parser.add_argument(
        'autoencoder',
        metavar='FILE',
        help='autoencoder config file;'
    )

    regressor_parser.add_argument(
        'model',
        metavar='DIR',
        nargs='?',
        help='output model directory;',
        default='models'
    )

    regressor_parser.add_argument(
        'experiment',
        metavar='DIR',
        nargs='?',
        help='output experiment directory;',
        default='experiments'
    )

    regressor_parser.add_argument(
        '-e', '--epochs',
        metavar='NUM',
        type=int,
        required=False,
        help='set maximum number of training epochs;',
        default=100
    )
    regressor_parser.add_argument(
        '-b', '--batch-size',
        metavar='NUM',
        type=int,
        required=False,
        help='set batch size;',
        default=32
    )
    regressor_parser.add_argument(
        '--max-smiles-length',
        metavar='NUM',
        type=int,
        required=False,
        help='set maximum SMILES length;',
        default=150
    )
    regressor_parser.add_argument(
        '--in-memory',
        action='store_const',
        const=True,
        help='set computation in memory;',
        default=False
    )

    regressor_parser.add_argument(
        '-v', '--verbose',
        action='store_const',
        const=True,
        help='set verbose mode;',
        default=False
    )

    # -----------------------------------------------------------
    # Defines Commands Actions.
    # -----------------------------------------------------------

    args = parser.parse_args(argv)

    if args.entity == 'autoencoder':
        print '\nTraining of autoencoder...\n'

        D_train, D_val, D_eval = _get_datasets(
            args.dataset, True, args.in_memory
        )

        model = create_autoencoder(smiles(args.max_smiles_length))
        model.summary()

        tmp = tempfile.gettempdir()
        print 'Checkpoints location -> %s' % tmp

        experiment = run_single_validation(
            model,
            D_train, D_val, D_eval,
            epochs=args.epochs, batch_size=args.batch_size,
            checkpoint_path=tmp,
            verbose=args.verbose
        )

    elif args.entity == 'classifier':
        print '\nTraining of classifier...\n'

        D_train, D_val, D_eval = _get_datasets(
            args.dataset, False, args.in_memory
        )

        tmp = tempfile.gettempdir()
        print 'Checkpoints path -> %s' % tmp

        autoencoder_path = os.path.dirname(args.autoencoder)
        autoencoder_base = os.path.basename(args.autoencoder)
        autoencoder_name, _ = os.path.splitext(autoencoder_base)

        print 'Autoencoder path -> %s' % autoencoder_path
        print 'Autoencoder name -> %s' % autoencoder_name

        model = create_classifier(
            category(['0', '1']),
            autoencoder_path=autoencoder_path,
            autoencoder_name=autoencoder_name)
        model.summary()

        experiment = run_single_validation(
            model,
            D_train, D_val, D_eval,
            epochs=args.epochs, batch_size=args.batch_size,
            checkpoint_path=tmp,
            verbose=int(args.verbose)
        )

    elif args.entity == 'regressor':
        print '\nTraining of regressor...\n'

        D_train, D_val, D_eval = _get_datasets(
            args.dataset, False, args.in_memory
        )

        tmp = tempfile.gettempdir()
        print 'Checkpoints path -> %s' % tmp

        autoencoder_path = os.path.dirname(args.autoencoder)
        autoencoder_base = os.path.basename(args.autoencoder)
        autoencoder_name, _ = os.path.splitext(autoencoder_base)

        print 'Autoencoder path -> %s' % autoencoder_path
        print 'Autoencoder name -> %s' % autoencoder_name

        model = create_regressor(
            numeric(-20, 20),
            autoencoder_path=autoencoder_path,
            autoencoder_name=autoencoder_name)
        model.summary()

        experiment = run_single_validation(
            model,
            D_train, D_val, D_eval,
            epochs=args.epochs, batch_size=args.batch_size,
            checkpoint_path=tmp,
            verbose=args.verbose
        )

    else:
        raise AssertionError('Unsupported entity: %s' % args.entity)

    # Defines timestamp for model and experiment files.
    base = os.path.basename(args.dataset)
    filename, filext = os.path.splitext(base)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Defines experiment name.
    experiment_name = '%s-%s-%s' % (
        filename,
        args.entity,
        timestamp
    )
    print 'Experiment name: %s' % experiment_name

    # Serializing the trained model.
    print 'Saving model...'
    if not os.path.exists(args.model):
        os.mkdir(args.model)
    model.save(args.model, experiment_name)

    # Serializing the experiment.
    print 'Saving experiment...'
    if not os.path.exists(args.experiment):
        os.mkdir(args.experiment)
    save_object(
        os.path.join(
            args.experiment,
            '%s.exp' % experiment_name
        ),
        experiment
    )

    # Show the summary of the model training.
    print_summary(experiment)


def runCmdPredict(argv):
    """Makes a prediction using a model."""
    parser = argparse.ArgumentParser(
        prog='predict', description='Makes a prediction using a model.'
    )

    # -----------------------------------------------------------
    # Defines command-line options for train a regressor.
    # -----------------------------------------------------------

    parser.add_argument(
        'model',
        metavar='FILE',
        help='model config file;'
    )

    parser.add_argument(
        'query',
        metavar='SMILES',
        help='SMILES representation of a chemical compound;'
    )

    # -----------------------------------------------------------
    # Defines Commands Actions.
    # -----------------------------------------------------------

    args = parser.parse_args(argv)

    dnn_path = os.path.dirname(args.model)
    dnn_base = os.path.basename(args.model)
    dnn_name, _ = os.path.splitext(dnn_base)

    print 'DNN path -> %s' % dnn_path
    print 'DNN name -> %s' % dnn_name

    print 'Loading deep learning neural network...'
    dnn = load_network(dnn_path, dnn_name)

    # Gets an estimator.
    estimator = create_estimator(dnn)

    # Transforms a query
    print estimator.predict(args.query)


def runCmdDescribe(argv):
    """Describes an experiment results."""
    parser = argparse.ArgumentParser(
        prog='describe', description='Describe an experiment.'
    )

    parser.add_argument(
        'experiment',
        metavar='FILE',
        help='experiment file name;'
    )

    args = parser.parse_args(argv)

    print '\nExperiment description...\n'
    exp = load_object(args.experiment)
    print_summary(exp)


def _get_datasets(path, mirror, in_memory):
    D_train, D_val, D_eval = None, None, None

    if in_memory:
        D = create_dataset(
            [record for record in read_records(path)],
            mirror
        )
        D_train, D_val, D_eval = D.partition(mode='tve')
    else:
        P_train, P_val, P_eval = split_dataset(path)
        D_train = create_dataset(P_train, mirror)
        D_val = create_dataset(P_val, mirror)
        D_eval = create_dataset(P_eval, mirror)

    return D_train, D_val, D_eval


# =====================================================
# Private Classes
# =====================================================

class ShellException(Exception):
    pass

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    try:
        cmd = None
        if len(sys.argv) > 1:
            cmd = sys.argv[1]
            if cmd == 'train':
                runCmdTrain(sys.argv[2:])
            elif cmd == 'predict':
                runCmdPredict(sys.argv[2:])
            elif cmd == 'describe':
                runCmdDescribe(sys.argv[2:])
            else:
                raise ShellException("Invalid command.")
        else:
            raise ShellException("Command has not defined.")
        exit(0)
    except ShellException as exc:
        print exc
        exit(1)
