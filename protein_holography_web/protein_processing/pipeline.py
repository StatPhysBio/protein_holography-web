
from .structural_info import get_structural_info
from .neighborhoods import get_neighborhoods
from .zernikegrams import get_zernikegrams

from typing import *


def get_zernikegrams_from_pdbfile(pdbfile: str,
                                     get_structural_info_kwargs: Dict,
                                     get_neighborhoods_kwargs: Dict,
                                     get_zernikegrams_kwargs: str):

    proteins = get_structural_info(pdbfile, **get_structural_info_kwargs)
    
    neighborhoods = get_neighborhoods(proteins, **get_neighborhoods_kwargs)

    zernikegrams = get_zernikegrams(neighborhoods, **get_zernikegrams_kwargs)

    return zernikegrams



