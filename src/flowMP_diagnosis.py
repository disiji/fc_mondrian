import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut


def LOO(X, Y):
    loo = LeaveOneOut()
    models = []
    X = np.array(X)
    Y = np.array(Y)

    predict_prob = []
    for train, test in loo.split(X, Y):
        train_X = X[train]
        train_Y = Y[train]
        test_X = X[test]
        test_Y = Y[test]
        logreg = linear_model.LogisticRegression(C=1e5)
        logreg.fit(train_X, train_Y)
        test_Y_predict = logreg.predict(test_X)
        models.append(logreg)
        predict_prob.append(logreg.predict_proba(test_X)[0][0])

    print(predict_prob)
    plt.scatter(range(21), predict_prob, s=100)
    plt.xlim(0, 21)
    plt.ylim(0, 1)
    groups = ['H%s' % i for i in range(1, 6)] + ['SJ%s' % i for i in range(1, 17)]
    plt.legend()

    plt.xticks(range(21), groups)
    plt.ylabel('P(healthy)')
    plt.title('P(healthy) Predicted by LOOCV Logistic Regression')

    return predict_prob, models
