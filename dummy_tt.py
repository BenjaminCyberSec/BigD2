import numpy as np

n_test = 1827
ham_ratio = 1721 / (779 + 1721)

fd = open('spam-mail.tt.label', 'w')

fd.write('Id,Prediction')

for email_id in range(1, n_test+1):
    ham_or_spam =  np.random.binomial(1, ham_ratio, size=(1))[0]

    fd.write('\n%d,%d' % (email_id, ham_or_spam))

fd.close()
