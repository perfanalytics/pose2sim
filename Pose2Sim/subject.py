#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SUBJECT                             ##
#########################################
'''

from Pose2Sim.config import SubConfig

class Subject:
    def __init__(self, config: SubConfig, data: dict):
        self.height = data.get("height", config.default_height)
        self.mass = data.get("mass", config.default_height)
