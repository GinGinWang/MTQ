
# normalize llk and rec
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def compute_threshold(self, cl):

		dataset = self.dataset.valid_split

        loader = DataLoader(dataset)

        sample_score = np.zeros(shape=(len(loader),))
        
        for i, (x, y) in enumerate(loader):
            x = x.to('cuda')
            x_r, z, z_dist, s, log_jacob_s = self.model(x)
            self.loss(x, x_r, z, z_dist,s, log_jacob_s)
            sample_score[i] = - self.loss.total_loss # large score -- normal


    best_e = 0
    best_f = 0
    best_e_ = 0
    best_f_ = 0

    # real label y  normal 0,  novel 1

    # predict score  sample_score 
    # predict label y_hat
    minS = sample_score.min() - 0.1
    maxS = sample_score.max() + 0.1

    for e in np.arange(minP, maxP, 0.1):

        y_hat = np.less(sample_score, e) #  normal 0  novel1
        # # TP Predict novel as novel y =1, y_hat =1
        # true_positive = np.sum(np.logical_and(y_hat, y))
        # # FP Predict normal as novel y = 0, y_hat = 1
        # false_positive = np.sum(np.logical_and(y_hat, logical_not(y)))
        # # PN Predict novel as normal y =1, y_hat = 0
        # false_negative = np.sum(np.logical_and(np.logical_not(y_hat),y))
        if true_positive > 0:
            f1 = f1_score(y, y_hat)
            recall = recall_score(y, y_hat)
            precision =  precision_score(y, y_hat)
          
            if f > best_f:
                best_f = f
                best_e = e
            if f >= best_f_:
                best_f_ = f
                best_e_ = e

    best_e = (best_e + best_e_) / 2.0

    print("Best e: ", best_e)
    return best_e
