import socket
import time
import datetime
import pycrayon
import numpy as np

from collections import namedtuple

CrayonSettings = namedtuple('CrayonSettings', ['host', 'port'])
CRAYON_SETTINGS = CrayonSettings(host='localhost', port='9119')


def get_experiment(name, settings=CRAYON_SETTINGS):
    """Creates a pycrayon experiment object to log data to."""
    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
    experiment_name = '{dt}_{host};{name}'.format(
        dt=experiment_date,
        host=socket.gethostname(),
        name=name,
    )
    return get_crayon_client(settings=settings).create_experiment(experiment_name)


def get_crayon_client(settings=CRAYON_SETTINGS):
    return pycrayon.CrayonClient(hostname=settings.host, port=settings.port)


def clear_expts(settings=CRAYON_SETTINGS):
	get_crayon_client(settings=settings).remove_all_experiments()