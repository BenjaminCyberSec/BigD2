# -*- coding: utf-8 -*-

def print_output(pred):
	
	fd = open('spam-mail.tt.label', 'w')
	fd.write('Id,Prediction')
	for email_id, ham_or_spam in zip(range(0, len(pred)), pred):
	    fd.write('\n%d,%d' % (email_id+1, ham_or_spam)) #+1 so that we start at 1
	fd.close()
		
		
