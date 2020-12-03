import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
import ss_custom

def plot_calibration_curve(est, name, fig_index,x_train, y_train,x_test,y_test,method_test,colour):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    #isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    #sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

     # Logistic regression with no calibration as baseline
    #lr = LogisticRegression(C=1., solver='lbfgs')

    fig_rel = plt.figure(fig_index, figsize=(4,5))
    ax_rel = plt.subplot2grid((4, 1), (0, 0), rowspan=3)

    ax_sharp = plt.subplot2grid((4, 1), (3, 0))

    ax_rel.plot([0, 1], [0, 1], "k:")
    est.fit(x_train, y_train)
    y_pred = est.predict(x_test)
    if hasattr(est, "predict_proba"):
        prob_pos = est.predict_proba(x_test)[:, 1]
    else:  # use decision function
        prob_pos = est.decision_function(x_test)
        prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    est_score = ss_custom.bss_calc(y_test, prob_pos)
    print("%s:" % name)
    print("\tBrier: %1.3f" % (est_score))
    print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
    print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

    fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

    ax_rel.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name), color=colour)

    ax_sharp.hist(prob_pos, range=(0, 1), bins=10,
                 histtype="step", lw=2,color='k')

    ax_rel.set_ylabel("Observed Frequency")
    ax_rel.set_ylim([-0.05, 1.05])
    ax_rel.legend(loc="lower right")
    ax_rel.set_title('Reliability Diagram')
    ax_rel.text(0.77, 0.1,'BSS=%.2f'%est_score)
    ax_sharp.set_xlabel("Forecast Probability")
    ax_sharp.set_ylabel("Count")
    #ax_sharp.legend(loc="upper center", ncol=2)

    plt.tight_layout()
   # plt.show()
    #plt.savefig(name+'_calibration_curve_'+method_test+'.eps')
    #plt.close()

#plot_calibration_curve(GaussianNB(), "Naive Bayes", \
                    #   1, X_train_split, Y_train_split, X_test_split, Y_test_split)
