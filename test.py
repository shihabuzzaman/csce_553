#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:30:36 2022

@author: shihab
"""
import unittest
from csce_553 import sumforline, readcsv, calculate_node_class

class unit_test(unittest.TestCase):
      
    def test_sumforline(self):
        self.assertEquals(sumforline("sample_1.csv"), 12)
        self.assertEquals(sumforline("split_0.csv"), 1000001)
        self.assertEquals(sumforline("split_1.csv"), 10001)
        
    def test_readcsv(self):
        self.assertEquals(readcsv("sample_1.csv", 11), 39)
        self.assertEquals(readcsv("split_0.csv", 100), 244)
        self.assertEquals(readcsv("split_1.csv", 100), 7)
        
    def test_calculate_node_class(self):
        self.assertEquals(calculate_node_class(12), 12)
        

        
      
if __name__ == '__main__':
   unittest.main()