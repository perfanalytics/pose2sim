#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SUBJECT                             ##
#########################################
'''

class Subject:
    def __init__(self, config, data: dict):
        self.height = data.get("height", config.default_height)
        self.mass = data.get("mass", config.default_height)
