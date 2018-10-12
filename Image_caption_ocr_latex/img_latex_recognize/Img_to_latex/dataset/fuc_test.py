#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-18 14:33:52
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-18 14:33:52

import re

str = '1-2-3+4+5-6-7+8+9-10-11+\cdots+2012+2013-2014-2015='
str2='\frac{x}{x+1}+\frac{1}{x-1} =1'
lis_str = list(str2)
print(lis_str)
patt = re.compile(r'.*\d+.*')
a = re.match(patt, str)
print(a)
b = re.search('.*([0-9]+).*', str)
print(b)
c = re.findall('[0-9]+', str)
print(c)
