try:
    from .bigearthnet_dataset import *
    from .eurosat_dataset import *
except ImportError as e:
    print(e)
    print("Please install the required packages for the BigEarthNet and EuroSAT datasets.")
from .imagenet_dataset import *
from .meter_dataset import *
from .satlas_dataset import *
from .fm4cs_dataset import *
from .fm4cs_dataset_v2 import *
from .fm4cs_iterable_dataset_v2 import *
