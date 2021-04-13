# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:30:51 2021

@author: Benjamin
"""

import numpy as np

def print_output(pred):
	
	fd = open('spam-mail.tt.label', 'w')
	fd.write('Id,Prediction')
	for email_id, ham_or_spam in zip(range(0, len(pred)), pred):
	    fd.write('\n%d,%d' % (email_id+1, ham_or_spam)) #+1 so that we start at 1
	fd.close()
		
		
