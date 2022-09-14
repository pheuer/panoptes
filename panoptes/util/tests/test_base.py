# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:31:08 2022

@author: pvheu
"""

import os

from panoptes.util.base import BaseObject



if __name__ == '__main__':
    
    tmp_path = os.path.join(os.getcwd(), 'base1.h5')
    
    obj = BaseObject(path=tmp_path)
    obj.save()

    obj2 = BaseObject(path=tmp_path)
