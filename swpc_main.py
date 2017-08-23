import shelve
from collections import OrderedDict
from itertools import cycle, islice

from sklearn.metrics import roc_curve, auc
import matplotlib.pylab as plt
import numpy
import pandas
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, \
    LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from f_kmeans_ import FKMeans

out_folder = './out/'


# from sknn.mlp import Classifier as MLPerceptron
# from sknn.mlp import Layer

def hybrid_false(estimator_, X_training, X_testing):
    Y_training_classification = estimator_.predict(X_training.as_matrix())
    Y_testing_classification = estimator_.predict(X_testing.as_matrix())

    return Y_training_classification, Y_testing_classification


def hybrid_true(estimator_, X_training, X_testing, fuzzy):
    Y_training_prediction = estimator_.predict_proba(X_training.as_matrix())

    threshold = clustering_classifier(Y_training_prediction[:, 1], fuzzy=fuzzy)

    Y_testing_prediction = estimator_.predict_proba(X_testing.as_matrix())
    Y_testing_classification = Y_testing_prediction[:, 1] > threshold
    Y_training_classification = Y_training_prediction[:, 1] > threshold

    return Y_training_classification, Y_testing_classification


def creation_roc(Y_actual, Y_predicted):
    fpr, tpr, thresholds = roc_curve(Y_actual, Y_predicted)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def check_prediction_clustering(Y_training, labels):
    labels_01 = numpy.zeros((labels.shape))
    labels_01[labels == 0] = 1

    classification_1 = classification_skills(Y_training, labels)
    classification_2 = classification_skills(Y_training, labels_01)

    # done with tss
    if classification_1[5] > classification_2[5]:
        labels_ok = labels
    else:
        labels_ok = labels_01

    return labels_ok


def classification_skills(y_real, y_pred):
    FP_rate, TP_rate, roc_auc_value = creation_roc(y_real, y_pred)
    # FP_rate = 1- SPC (1- TN(TN+FP))
    # TP_rate = POD (sensitivity TP/(TP+FN)

    cm = confusion_matrix(y_real, y_pred)
    a = float(cm[1, 1]) + 1e-10
    d = float(cm[0, 0]) + 1e-10
    b = float(cm[0, 1]) + 1e-10
    c = float(cm[1, 0]) + 1e-10
    TP = a
    TN = d
    FP = b
    FN = c
    hss = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    tss = (TP / (TP + FN)) - (FP / (FP + TN))
    fnfp = FN / FP
    pod = TP / (TP + FN)
    far = FP / (TP + FP)
    acc = (TP + TN) / (TP + FP + FN + TN)
    J = pod + TN / (TN + FP) - 1
    '''
    print ('confusion matrix')
    print (cm)
    print ('false alarm ratio       \t', far)
    print ('probability of detection\t', pod)
    print ('accuracy                \t', acc)
    print ('hss                     \t', hss)
    print ('tss                     \t', tss)
    print ('balance                 \t', fnfp)
    print ('J                       \t', J)
    '''
    return cm.tolist(), far, pod, acc, hss, tss, fnfp, J, FP_rate, TP_rate, roc_auc_value



def load_data(freq_class):
    X_training = pandas.read_pickle('data/X_training_' + freq_class + '.pkl')
    Y_training = pandas.read_pickle('data/Y_training_' + freq_class + '.pkl')
    X_testing = pandas.read_pickle('data/X_testing_' + freq_class + '.pkl')
    Y_testing = pandas.read_pickle('data/Y_testing_' + freq_class + '.pkl')

    return X_training, Y_training, X_testing, Y_testing


### ALGORITHMS

def randomForest(X_training, Y_training, X_testing, Y_testing, hybrid=False, fuzzy=True):
    estimator_ = RandomForestClassifier(n_estimators=50,
                                        criterion='gini',
                                        max_depth=None,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_features='auto',
                                        max_leaf_nodes=None,
                                        min_impurity_split=0.000001,
                                        bootstrap=True,
                                        oob_score=False,
                                        n_jobs=1,
                                        random_state=None,
                                        verbose=0,
                                        warm_start=False,
                                        class_weight=None)

    estimator_.fit(X_training.as_matrix(), Y_training.as_matrix())

    if hybrid == False:
        print('RF')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_false(estimator_, X_training, X_testing)

    elif hybrid == True:
        print('RF + clustering')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_true(estimator_, X_training, X_testing, fuzzy)

    feat_importance = estimator_.feature_importances_
    print('MtWilson', 'Z', 'p', 'c', 'Area')
    print feat_importance

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification), \
           feat_importance


def Kmeans(X_training, Y_training, X_testing, Y_testing):
    print('C-Means')
    print('--------------')

    estimator_ = KMeans(n_clusters=2,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        tol=0.0001,
                        algorithm='full',
                        precompute_distances='auto',
                        verbose=0,
                        random_state=None,
                        copy_x=True,
                        n_jobs=1)

    estimator_.fit(X_training.as_matrix(), Y_training.as_matrix())

    Y_training_classification = check_prediction_clustering(Y_training, estimator_.labels_)

    Y_testing_classification = estimator_.predict(X_testing.as_matrix())

    Y_testing_classification = check_prediction_clustering(Y_testing, Y_testing_classification)

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification)


def Fkmeans(X_training, Y_training, X_testing, Y_testing):
    print('Fuzzy C-Means')
    print('--------------')

    estimator_ = FKMeans(n_clusters=2,
                         distance='euclidean',
                         m=2.0,
                         tol_centroids=1e-4,
                         tol_memberships=1e-4,
                         max_iter=1000,
                         n_init=10,
                         constraint='probabilistic')

    estimator_.fit(X_training.as_matrix(), Y_training.as_matrix())

    Y_training_classification = check_prediction_clustering(Y_training, estimator_.labels_)

    Y_testing_classification = estimator_.predict(X_testing.as_matrix())

    Y_testing_classification = check_prediction_clustering(Y_testing, Y_testing_classification)

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification)


def multiLayerPerceptron(X_training, Y_training, X_testing, Y_testing, hybrid=False, fuzzy=True):
    estimator_ = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(50, 50),
                               activation='relu',
                               random_state=1, verbose=False)
    estimator_.fit(X_training.as_matrix(), Y_training.as_matrix())

    if hybrid == False:
        print('mlp')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_false(estimator_, X_training, X_testing)
    elif hybrid == True:
        print('mlp + clustering')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_true(estimator_, X_training, X_testing, fuzzy)

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification)


def supportVectorMachine(X_training, Y_training, X_testing, Y_testing, hybrid=False, fuzzy=True):
    estimator_ = SVC(C=1.0, probability=True)
    estimator_.fit(X_training.as_matrix(), Y_training.as_matrix())
    if hybrid == False:
        print('svm')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_false(estimator_, X_training, X_testing)

    elif hybrid == True:
        print('svm + clustering')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_true(estimator_, X_training, X_testing, fuzzy)

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification)


def logit(X_training, Y_training, X_testing, Y_testing, hybrid=False, fuzzy=True):
    estimator_ = LogisticRegressionCV(Cs=numpy.logspace(-4, 4, 100),
                                      fit_intercept=True,
                                      cv=3,
                                      dual=False,
                                      penalty='l1',
                                      solver='liblinear',
                                      tol=0.0001,
                                      max_iter=1000,
                                      class_weight=None,
                                      n_jobs=1,
                                      verbose=0,
                                      refit=True,
                                      intercept_scaling=1.0,
                                      multi_class='ovr',
                                      random_state=None)
    estimator_.fit(X_training, Y_training)

    if hybrid == False:
        print('logit')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_false(estimator_, X_training, X_testing)

    elif hybrid == True:
        print('logit + clustering')
        print('--------------')
        Y_training_classification, Y_testing_classification = hybrid_true(estimator_, X_training, X_testing, fuzzy)

    feat_importance = (estimator_.coef_ / estimator_.coef_.sum())[0]
    print('MtWilson', 'Z', 'p', 'c', 'Area')
    print feat_importance

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification), \
           feat_importance


def hybrid(X_training, Y_training, X_testing, Y_testing, hybrid=True, fuzzy=True):
    print 'hybrid method: lasso + kmeans'
    print '--------------'
    estimator_ = LassoCV(eps=0.001,
                         n_alphas=100,
                         alphas=None,
                         fit_intercept=True,
                         normalize=False,  # True
                         precompute='auto',
                         max_iter=1000,
                         tol=0.0001,
                         cv=3,
                         verbose=False,
                         n_jobs=-1,
                         positive=True,
                         random_state=None,
                         selection='cyclic')
    estimator_.fit(X_training, Y_training)
    Y_training_prediction, Y_testing_prediction = hybrid_false(estimator_, X_training, X_testing)

    if hybrid == False:
        print('lasso')
        print('--------------')
        threshold = 0.5

    elif hybrid == True:
        print('hybrid')
        print('--------------')
        threshold = clustering_classifier(Y_training_prediction, fuzzy)

    print threshold

    Y_training_classification = Y_training_prediction > threshold
    Y_testing_classification = Y_testing_prediction > threshold

    feat_importance = estimator_.coef_ / estimator_.coef_.sum()
    print('MtWilson', 'Z', 'p', 'c', 'Area')
    print feat_importance

    return classification_skills(Y_training, Y_training_classification), \
           classification_skills(Y_testing, Y_testing_classification), \
           feat_importance


def clustering_classifier(Y_training_prediction, fuzzy=True):
    if fuzzy == True:
        est_ = FKMeans(n_clusters=2,
                       distance='euclidean',
                       constraint='probabilistic',
                       m=2.,
                       tol_centroids=1e-6,
                       tol_memberships=1e-6,
                       max_iter=10000,
                       n_init=150)
    elif fuzzy == False:
        est_ = KMeans(n_clusters=2,
                      init='k-means++',
                      n_init=10,
                      max_iter=300,
                      tol=0.0001,
                      precompute_distances='auto',
                      verbose=0,
                      random_state=None,
                      copy_x=True, n_jobs=1,
                      algorithm='auto')

    abscissa = Y_training_prediction.reshape(-1, 1)
    est_.fit(abscissa)
    classes = est_.labels_

    # flare / no flare predicted via regression
    no_flare_abscissa = abscissa[classes == 0]
    flare_abscissa = abscissa[classes == 1]
    how_many_no_flares = float(no_flare_abscissa.shape[0])
    how_many_flares = float(flare_abscissa.shape[0])
    flare_rate = how_many_no_flares / how_many_flares
    if flare_rate < 1.:
        no_flare_abscissa = abscissa[classes == 1]
        flare_abscissa = abscissa[classes == 0]
        how_many_no_flares = float(no_flare_abscissa.shape[0])
        how_many_flares = float(flare_abscissa.shape[0])
        flare_rate = how_many_no_flares / how_many_flares

    threshold = (max(no_flare_abscissa) + min(flare_abscissa)) / 2.

    print 'threshold\t', threshold

    return threshold


def save_skills(skills, flare_type, data_type, alg_name):
    skills_dict = {'hss': skills[4],
                   'tss': skills[5],
                   'pod': skills[2],
                   'acc': skills[3],
                   'far': skills[1],
                   'fn/fp': skills[6],
                   'J': skills[7],
                   'cm': skills[0],
                   'FP_rate': skills[8],
                   'TP_rate': skills[9],
                   'roc_auc_value': skills[10]}

    new_dict = {alg_name: {flare_type: skills_dict}}

    # saving on file-database
    db = shelve.open(out_folder + 'swpc-results_' + data_type + '_' + flare_type + '_class')
    db.update(new_dict)
    db.close()


def print_swpc_feature_importance_table(logit_feat_importance,
                                        hybrid_feat_importance, RF_feat_importance):
    m = numpy.array([logit_feat_importance,
                     hybrid_feat_importance, RF_feat_importance])

    df = DataFrame(m, index=['logit', 'hybrid', 'RF'],
                   columns=['MtWilson', 'Z', 'p', 'c', 'Area'])

    print df.to_latex()

    print(logit_feat_importance)
    print(hybrid_feat_importance)
    print(RF_feat_importance)


def select_flares(which_class, Y_training, Y_testing):
    print('--------------')
    print(which_class, 'flares ')
    print('--------------\n')

    if which_class == 'C':
        # C flares
        Y_training_labels = Y_training['C1']
        Y_testing_labels = Y_testing['C1']

    if which_class == 'M':
        # M flares
        Y_training_labels = Y_training['M1']  # & Y_training['C1']
        Y_testing_labels = Y_testing['M1']  # & Y_testing['C1']

    if which_class == 'X':
        # X flares
        Y_training_labels = Y_training['X1']  # & Y_training['M1'] & Y_training['C1']
        Y_testing_labels = Y_testing['X1']  # & Y_testing['M1'] & Y_testing['C1']

    return Y_training_labels, Y_testing_labels


def normalization(X_training, Y_training, X_testing, Y_testing):
    # normalization / standardization
    mean_ = X_training.sum(axis=0) / X_training.shape[0]
    std_ = numpy.sqrt(((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0])
    # Xn_training = scale(X_training)
    Xn_training = (X_training - mean_) / std_
    # Xn_testing
    Xn_testing = (X_testing - mean_) / std_

    mean_Y_ = Y_training.sum(axis=0) / Y_training.shape[0]
    std_Y_ = numpy.sqrt(((Y_training - mean_Y_) ** 2.).sum(axis=0) / Y_training.shape[0])
    # Xn_training = scale(X_training)
    Yn_training = (Y_training - mean_Y_) / std_Y_
    # Xn_testing
    Yn_testing = (Y_testing - mean_Y_) / std_Y_

    return Xn_training, Yn_training, Xn_testing, Yn_testing


def swpc_histograms(flare_class, data_type):
    plt.close("all")
    db = shelve.open(out_folder + 'swpc-results_' + data_type + '_' + flare_class + '_class')

    skills = ['FAR ', 'ACC ', 'POD ', 'TSS ', 'HSS ', 'J']

    name_ = OrderedDict()

    # name_['Lasso'] = 'Lasso'
    # name_['l1-logit Hybrid'] = 'l1-logit Hybrid'
    # name_['Hybrid Random Forest'] = 'Hybrid Random Forest'
    name_['Random Forest'] = 'Random Forest'
    name_['Fuzzy C-Means'] = 'Fuzzy C-Means'
    name_['K-Means'] = 'K-Means'
    name_['Hybrid'] = 'Hybrid'
    name_['l1-logit'] = 'l1-logit'
    name_['Multi Layer Perceptron'] = 'Multi Layer Perceptron'
    name_['Support Vector Machine'] = 'Support Vector Machine'

    m = []
    m_roc = []
    algs = []
    for alg in name_.keys():  # sorted(db.keys()):
        print(alg)
        algs.append(name_[alg])
        s = db[alg][flare_class]
        skill_vector = [s['tss'], s['hss'], s['acc'], s['pod'], s['far']]
        roc_vector = [s['FP_rate'], s['TP_rate'], s['roc_auc_value']]
        m.append(skill_vector)
        m_roc.append(roc_vector)
        print(alg)
        print(skill_vector)
    db.close()

    m = numpy.array(m).transpose()

    skills = ['TSS ', 'HSS ', 'ACC ', 'Sensitivity', 'FAR ']
    # our_colors = ['black','grey', 'lightgray',
    our_colors = ['hotpink', 'darkslateblue', 'cornflowerblue', 'orangered', 'orange', 'gold', 'mediumseagreen']
    my_colors = list(islice(cycle(our_colors), None, 24))
    # my_colors.reverse()
    # alg.reverse()

    print(m)

    df = DataFrame(m, columns=algs, index=skills)
    print(df)
    df = df.astype(float)
    df.plot(kind='barh', fontsize=14, figsize=[10, 8], color=my_colors,
            width=0.7, xlim=[0, 1.])  # legend = 'reverse',
    plt.xlabel('Skill score values ', fontsize=16)
    plt.title('Skill comparison (' + flare_class + ' class flare)', fontsize=16)
    mt = numpy.arange(0, 1.1, 0.25)
    plt.xticks(mt)
    plt.grid(b=True, axis='x')
    plt.savefig(out_folder + 'skill_comparison_' + data_type + '_' + flare_class + '_class.png', format="png")
    plt.close()

    '''
    # plot ROC curve
    m_roc_array = numpy.array(m_roc)
    plt.figure()
    plt.title('Receiver Operating Characteristic ')

    for it in numpy.arange(m_roc_array.shape[0]):
        plt.plot(m_roc_array[it][0], m_roc_array[it][1], my_colors[it],
                 label=algs[it] + ' AUC = %0.2f' % m_roc_array[it][2])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig(out_folder + 'ROC_' + data_type + '_' + flare_class + '_class.png', format="png")
    plt.close()
    '''

if __name__ == '__main__':

    # use <freq_class> for predicting <flare_class>
    freq_class = 'C'
    flare_class = 'C'

    X_training, Y_training, X_testing, Y_testing = \
        load_data( freq_class)


    Y_training_labels, Y_testing_labels = \
        select_flares(flare_class, Y_training, Y_testing)

    Xn_training, Yn_training_labels, Xn_testing, Yn_testing_labels = normalization(X_training, Y_training_labels,
                                                                                   X_testing, Y_testing_labels)



    '''`
    rf
    '''
    RF_skills = randomForest(Xn_training, Y_training_labels,
              Xn_testing, Y_testing_labels, hybrid=False,fuzzy=True)

    save_skills(RF_skills[0], flare_class,'training', 'Random Forest')
    save_skills(RF_skills[1], flare_class, 'testing', 'Random Forest')

    '''
    logistic regression
    '''
    logit_skills = logit(Xn_training, Y_training_labels,
                         Xn_testing, Y_testing_labels, hybrid=False,fuzzy=True)
    save_skills(logit_skills[0], flare_class, 'training', 'l1-logit')
    save_skills(logit_skills[1], flare_class, 'testing', 'l1-logit')


    '''
    lasso + fuzzy c means
    '''
    hybrid_skills =  hybrid(Xn_training, Y_training_labels,
                            Xn_testing,  Y_testing_labels, hybrid = True, fuzzy=True)
    save_skills(hybrid_skills[0], flare_class, 'training', 'Hybrid')
    save_skills(hybrid_skills[1], flare_class, 'testing', 'Hybrid')



    logit_feat_importance = logit_skills[2]
    hybrid_feat_importance = hybrid_skills[2]
    RF_feat_importance = RF_skills[2]

    print_swpc_feature_importance_table(logit_feat_importance,
                                        hybrid_feat_importance, RF_feat_importance)

    '''
    mlp
    '''
    mlp_skills = multiLayerPerceptron(Xn_training, Y_training_labels,
                                      Xn_testing,  Y_testing_labels,hybrid=False,fuzzy=True)
    save_skills(mlp_skills[0], flare_class, 'training', 'Multi Layer Perceptron')
    save_skills(mlp_skills[1], flare_class, 'testing', 'Multi Layer Perceptron')

    '''
    KMeans
    '''
    kmeans_skills = Kmeans(Xn_training, Y_training_labels, Xn_testing,  Y_testing_labels)
    save_skills(kmeans_skills[0], flare_class, 'training', 'K-Means')
    save_skills(kmeans_skills[1], flare_class, 'testing',  'K-Means')

    '''
    svm
    '''
    svm_skills = supportVectorMachine(Xn_training, Y_training_labels,
                                      Xn_testing, Y_testing_labels, hybrid=False,fuzzy=True)
    save_skills(svm_skills[0], flare_class,'training', 'Support Vector Machine')
    save_skills(svm_skills[1], flare_class, 'testing', 'Support Vector Machine')

    '''
    FKMeans - probabilistic
    '''
    fkmeans_skills = Fkmeans(Xn_training, Y_training_labels,
                             Xn_testing, Y_testing_labels)
    save_skills(fkmeans_skills[0], flare_class, 'training', 'Fuzzy C-Means')
    save_skills(fkmeans_skills[1], flare_class, 'testing', 'Fuzzy C-Means')

    swpc_histograms(flare_class, 'training')
    swpc_histograms(flare_class, 'testing')


