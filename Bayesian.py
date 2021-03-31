import numpy as np
import pymc3 as mc
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import pandas as pd

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

# 各个学习参数的先验概率，alpha and beta
alpha0 = theano.shared(np.array([50, 80, 6, 40, 20, 5, 1, 90]))
beta0 = theano.shared(np.array([50, 20, 94, 60, 80, 95, 99, 10]))
alpha11 = theano.shared(np.array([90, 10, 50, 0.1, 9]))
beta11 = theano.shared(np.array([10, 90, 50, 99.9, 1]))
alpha12 = theano.shared(np.array([70, 2, 1, 0.1, 8]))
beta12 = theano.shared(np.array([30, 98, 99, 99.9, 2]))
alpha13 = theano.shared(np.array([8, 3, 3, .5]))
beta13 = theano.shared(np.array([992, 997, 997, 999.5]))
alpha14 = theano.shared(np.array([.6, .4, .2, .1, .1, .05, .03, .01]))
beta14 = theano.shared(np.array([999.4, 999.6, 999.8, 999.9, 999.9, 999.95, 999.97, 999.99]))
alpha2 = theano.shared(np.array([40, 10, 80, 50, 70, 20, 70, 40]))
beta2 = theano.shared(np.array([60, 90, 20, 50, 30, 80, 30, 60]))
alpha3 = theano.shared(np.array([99, .1]))
beta3 = theano.shared(np.array([1, 99.9]))
alpha4 = theano.shared(np.array([80, 20, 30, 70, 10]))
beta4 = theano.shared(np.array([20, 80, 70, 30, 90]))

sl_data = theano.shared(np.array([200, 200, 200, 200]))
pd_data = theano.shared(np.array([500, 500, 500, 500]))
dermoscopy_data = theano.shared(np.array([100, 100, 100, 100]))

with mc.Model() as model:
    # n为学习集总数
    n = 530

    # theta为第一层结点
    # observed为学习集中观测到的数据
    theta01 = mc.Beta('theta01', alpha=alpha0[0], beta=beta0[0])
    Age = mc.Binomial('Age', n=n, p=theta01, observed=[227])

    theta02 = mc.Beta('theta02', alpha=alpha0[1], beta=beta0[1])
    Site = mc.Binomial('Site', n=n, p=theta02, observed=[499])

    theta03 = mc.Beta('theta03', alpha=alpha0[2], beta=beta0[2])
    Susceptible_Factors = mc.Binomial('Factors', n=n, p=theta03, observed=[30])

    theta04 = mc.Beta('theta04', alpha=alpha0[3], beta=beta0[3])
    Light_Skin = mc.Binomial('LightSkin', n=n, p=theta04, observed=[200])

    theta05 = mc.Beta('theta05', alpha=alpha0[4], beta=beta0[4])
    Sunlight = mc.Binomial('Sunlight', n=n, p=theta05, observed=[120])

    theta06 = mc.Beta('theta06', alpha=alpha0[5], beta=beta0[5])
    Other = mc.Binomial('Other', n=n, p=theta06, observed=[20])

    theta07 = mc.Beta('theta07', alpha=alpha0[6], beta=beta0[6])
    Size = mc.Binomial('Size', n=n, p=theta07, observed=[181])

    theta08 = mc.Beta('theta08', alpha=alpha0[7], beta=beta0[7])
    Age2 = mc.Binomial('Age2', n=n, p=theta08, observed=[512])

    # 第二层
    rate11 = mc.Beta('rate11', alpha=alpha11, beta=beta11, shape=5)
    p11 = (rate11[0] * theta01 * theta02 + rate11[1] * theta01 * (1 - theta02) + rate11[2] * (1 - theta01) * theta02 +
           rate11[3] * (1 - theta01) * (1 - theta02)) + rate11[4]

    rate12 = mc.Beta('rate12', alpha=alpha12, beta=beta12, shape=5)
    p12 = (rate12[0] * theta08 * theta07 + rate12[1] * theta08 * (1 - theta07) + rate12[2] * (1 - theta08) * theta07 +
           rate12[3] * (1 - theta08) * (1 - theta07)) + rate12[4]

    rate13 = mc.Beta('rate13', alpha=alpha13, beta=beta13, shape=4)
    p13 = rate13[0] * theta03 * theta01 + rate13[1] * theta03 * (1 - theta01) + rate13[2] * (1 - theta03) * theta01 + \
          rate13[3] * (1 - theta03) * (1 - theta01)

    rate14 = mc.Beta('rate14', alpha=alpha14, beta=beta14, shape=8)
    p14 = rate14[0] * theta04 * theta05 * theta06 + rate14[1] * theta04 * theta05 * (1 - theta06) + rate14[
        2] * theta04 * (1 - theta05) * theta06 + rate14[3] * theta04 * (1 - theta05) * (1 - theta06) + rate14[4] * (
                      1 - theta04) * theta05 * theta06 + rate14[5] * (1 - theta04) * theta05 * theta06 + rate14[6] * (
                      1 - theta04) * (1 - theta05) * theta06 + rate14[7] * (1 - theta04) * (1 - theta05) * (1 - theta06)

    p_total = p11 + p12 + p13 + p14
    # theta11 = mc.Deterministic('theta11', tt.switch(tt.ge(p11, 1), .99, p11))
    theta11 = mc.Deterministic('theta11', p11 / p_total)
    Seborrheic_Kerratosis = mc.Binomial('SK', n=n, p=theta11, observed=[267])
    theta12 = mc.Deterministic('theta12', p12 / p_total)
    Acquired_Melanocytic_Nevi = mc.Binomial('AMN', n=n, p=theta12, observed=[255])
    theta13 = mc.Deterministic('theta13', p13 / p_total)
    Basal_cell_carcinoma = mc.Binomial('BCC', n=n, p=theta13, observed=[6])
    theta14 = mc.Deterministic('theta14', p14 / p_total)
    Melanoma = mc.Binomial('Mela', n=n, p=theta14, observed=[2])

    # 第三层
    rate2 = mc.Beta('rate2', alpha=alpha2, beta=beta2, shape=8)
    theta2 = [rate2[0] * theta11 + rate2[1] * (1 - theta11), rate2[2] * theta12 + rate2[3] * (1 - theta12),
              rate2[4] * theta13 + rate2[5] * (1 - theta13), rate2[6] * theta14 + rate2[7] * (1 - theta14)]
    Skin_Lesions = mc.Binomial('SL', n=[n, n, n, n], p=tt.as_tensor_variable(theta2), observed=[530, 530, 530, 530])

    rate3 = mc.Beta('rate3', alpha=alpha3, beta=beta3, shape=2)
    theta3 = [rate3[0] * theta11 + rate3[1] * (1 - theta11), rate3[0] * theta12 + rate3[1] * (1 - theta12),
              rate3[0] * theta13 + rate3[1] * (1 - theta13), rate3[0] * theta14 + rate3[1] * (1 - theta14)]
    Pathological_Description = mc.Binomial('PD', n=[n, n, n, n], p=tt.as_tensor_variable(theta3), shape=4)

    rate4 = mc.Beta('rate4', alpha=alpha4, beta=beta4, shape=5)
    theta4 = [rate4[0] * theta11 + rate4[1] * (1 - theta11), rate4[0] * theta12 + rate4[1] * (1 - theta12),
              rate4[0] * theta13 + rate4[2] * (1 - theta13), rate4[3] * theta14 + rate4[4] * (1 - theta14)]
    Dermoscopy = mc.Binomial('Dermoscopy', n=[n, n, n, n], p=tt.as_tensor_variable(theta4), shape=4)

with model:
    # if __name__ == '__main__':
    #     step = mc.Metropolis()
    #     Hives_trace = mc.sample(2000, step=step)
    #     mc.traceplot(Hives_trace)
    #     plt.show()

    # 检测RV集，校验用
    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))

    # 采样
    Hives_trace = mc.sample(draws=5000, tune=500, chains=1)
    # 数据整理
    print("rate11", Hives_trace['rate11'])
    print(np.mean(Hives_trace['rate11'], axis=0))
    r_sk = np.mean(Hives_trace['rate11'], axis=0)
    print("rate12", Hives_trace['rate12'])
    print(np.mean(Hives_trace['rate12'], axis=0))
    r_amn = np.mean(Hives_trace['rate12'], axis=0)
    r_bcc = np.mean(Hives_trace['rate13'], axis=0)
    r_mela = np.mean(Hives_trace['rate14'], axis=0)
    print("rate14", Hives_trace['rate14'])
    print(np.mean(Hives_trace['rate14'], axis=0))
    print("rate2", Hives_trace['rate2'])
    print(np.mean(Hives_trace['rate2'], axis=0))
    r_sl = np.mean(Hives_trace['rate2'], axis=0)
    print("rate3", Hives_trace['rate3'])
    print(np.mean(Hives_trace['rate3'], axis=0))
    r_pd = np.mean(Hives_trace['rate3'], axis=0)
    print("rate4", Hives_trace['rate4'])
    print(np.mean(Hives_trace['rate4'], axis=0))
    r_der = np.mean(Hives_trace['rate4'], axis=0)
    print(sum(Hives_trace['theta01']) / len(Hives_trace['theta01']),
          1 - sum(Hives_trace['theta01']) / len(Hives_trace['theta01']))
    p_age = sum(Hives_trace['theta01']) / len(Hives_trace['theta01'])
    print(sum(Hives_trace['theta02']) / len(Hives_trace['theta02']),
          1 - sum(Hives_trace['theta02']) / len(Hives_trace['theta02']))
    p_site = sum(Hives_trace['theta02']) / len(Hives_trace['theta02'])
    print(sum(Hives_trace['theta03']) / len(Hives_trace['theta03']),
          1 - sum(Hives_trace['theta03']) / len(Hives_trace['theta03']))
    p_factor = sum(Hives_trace['theta03']) / len(Hives_trace['theta03'])
    print(sum(Hives_trace['theta04']) / len(Hives_trace['theta04']),
          1 - sum(Hives_trace['theta04']) / len(Hives_trace['theta04']))
    p_light_skin = sum(Hives_trace['theta04']) / len(Hives_trace['theta04'])
    print(sum(Hives_trace['theta05']) / len(Hives_trace['theta05']),
          1 - sum(Hives_trace['theta05']) / len(Hives_trace['theta05']))
    p_sunlight = sum(Hives_trace['theta05']) / len(Hives_trace['theta05'])
    print(sum(Hives_trace['theta06']) / len(Hives_trace['theta06']),
          1 - sum(Hives_trace['theta06']) / len(Hives_trace['theta06']))
    p_other = sum(Hives_trace['theta06']) / len(Hives_trace['theta06'])
    print(sum(Hives_trace['theta07']) / len(Hives_trace['theta07']),
          1 - sum(Hives_trace['theta07']) / len(Hives_trace['theta07']))
    p_size = sum(Hives_trace['theta07']) / len(Hives_trace['theta07'])
    print(sum(Hives_trace['theta08']) / len(Hives_trace['theta08']),
          1 - sum(Hives_trace['theta08']) / len(Hives_trace['theta08']))
    p_age2 = sum(Hives_trace['theta08']) / len(Hives_trace['theta08'])
    print("SK:", sum(Hives_trace['theta11']) / len(Hives_trace['theta11']),
          1 - sum(Hives_trace['theta11']) / len(Hives_trace['theta11']))
    print("AMN:", sum(Hives_trace['theta12']) / len(Hives_trace['theta12']),
          1 - sum(Hives_trace['theta12']) / len(Hives_trace['theta12']))
    print("BCC:", sum(Hives_trace['theta13']) / len(Hives_trace['theta13']),
          1 - sum(Hives_trace['theta13']) / len(Hives_trace['theta13']))
    print("Mela:", sum(Hives_trace['theta14']) / len(Hives_trace['theta14']),
          1 - sum(Hives_trace['theta14']) / len(Hives_trace['theta14']))
    print(Hives_trace['theta02'])
    print("theta02:", sum(Hives_trace['theta02']) / len(Hives_trace['theta02']),
          1 - sum(Hives_trace['theta02']) / len(Hives_trace['theta02']))
    print(Hives_trace['theta01'])
    print("theta01:", sum(Hives_trace['theta01']) / len(Hives_trace['theta01']),
          1 - sum(Hives_trace['theta01']) / len(Hives_trace['theta01']))
    print("theta07:", sum(Hives_trace['theta07']) / len(Hives_trace['theta07']),
          1 - sum(Hives_trace['theta07']) / len(Hives_trace['theta07']))
    print("theta08:", sum(Hives_trace['theta08']) / len(Hives_trace['theta08']),
          1 - sum(Hives_trace['theta08']) / len(Hives_trace['theta08']))

# Model2，用作检测
print("model 2!!!")
sk_cpt = theano.shared(np.array([[r_sk[3], r_sk[2]], [r_sk[1], r_sk[0]]]))
sk_comp = theano.shared(r_sk[4])
amn_cpt = theano.shared(np.array([[r_amn[3], r_amn[2]], [r_amn[1], r_amn[0]]]))
amn_comp = theano.shared(r_amn[4])
bcc_cpt = theano.shared(np.array([[r_bcc[3], r_bcc[2]], [r_bcc[1], r_bcc[0]]]))
mela_cpt = theano.shared(np.array([[[r_mela[7], r_mela[6]], [r_mela[5], r_mela[4]]], [[r_mela[3], r_mela[2]], [r_mela[1], r_mela[0]]]]))
sl_cpt = theano.shared(np.array([[r_sl[1], r_sl[0]], [r_sl[3], r_sl[2]], [r_sl[5], r_sl[4]], [r_sl[7], r_sl[6]]]))
pd_cpt = theano.shared(np.array([r_pd[1], r_pd[0]]))
der_cpt = theano.shared(np.array([[r_der[1], r_der[0]], [r_der[1], r_der[0]], [r_der[2], r_der[0]], [r_der[4], r_der[3]]]))

with mc.Model() as model2:
    Age2 = mc.Bernoulli('Age', p=p_age, observed=0)
    Age22 = mc.Bernoulli('Age2', p=p_age2, observed=1)
    Site2 = mc.Bernoulli('Site', p=p_site, observed=1)
    Size2 = mc.Bernoulli('Size', p=p_size, observed=0)
    Factors = mc.Bernoulli('Factors', p=p_factor)
    Light_Skin2 = mc.Bernoulli('LightSkin', p=p_light_skin)
    Sunlight2 = mc.Bernoulli('SunLight', p=p_sunlight)
    Other2 = mc.Bernoulli('Other', p=p_other)

    p1 = sk_cpt[Age2, Site2] + sk_comp
    p2 = amn_cpt[Age22, Size2] + amn_comp
    p3 = bcc_cpt[Factors, Age2]
    p4 = mela_cpt[Light_Skin2, Sunlight2, Other2]
    p_total = p1 + p2 + p3 + p4
    Seborrheic_Kerratosis2 = mc.Bernoulli('SK', p=p1 / p_total)
    Acquired_Melanocytic_Nevi2 = mc.Bernoulli('AMN', p=p2 / p_total)
    Basal_cell_carcinoma2 = mc.Bernoulli('BCC', p=p3 / p_total)
    Melanoma2 = mc.Bernoulli('Mela', p=p4 / p_total)

    Skin_Lesions2 = mc.Bernoulli('SL', p=tt.as_tensor_variable(
        [sl_cpt[0][Seborrheic_Kerratosis2], sl_cpt[1][Acquired_Melanocytic_Nevi2], sl_cpt[2][Basal_cell_carcinoma2],
         sl_cpt[3][Melanoma2]]), shape=4, observed=[1, 1, 1, 1])
    Pathological_Description2 = mc.Bernoulli('PD', p=tt.as_tensor_variable(
        [pd_cpt[Seborrheic_Kerratosis2], pd_cpt[Acquired_Melanocytic_Nevi2], pd_cpt[Basal_cell_carcinoma2],
         pd_cpt[Melanoma2]]), shape=4)
    Dermoscopy2 = mc.Bernoulli('Der', p=tt.as_tensor_variable(
        [der_cpt[0][Seborrheic_Kerratosis2], der_cpt[1][Acquired_Melanocytic_Nevi2], der_cpt[2][Basal_cell_carcinoma2],
         der_cpt[3][Melanoma2]]), shape=4)

with model2:
    # if __name__ == '__main__':
    #     step = mc.Metropolis()
    #     Hives_trace = mc.sample(2000, step=step)
    #     mc.traceplot(Hives_trace)
    #     plt.show()
    Hives_trace = mc.sample(draws=5000, tune=500, chains=1)
    print("SK", Hives_trace['SK'])
    count1 = Counter(Hives_trace['SK'])
    true_times1 = count1[1]
    false_times1 = count1[0]
    print(true_times1 / float(false_times1 + true_times1))
    print("AMN", Hives_trace['AMN'])
    count2 = Counter(Hives_trace['AMN'])
    true_times2 = count2[1]
    false_times2 = count2[0]
    print(true_times2 / float(false_times2 + true_times2))
    print("BCC", Hives_trace['BCC'])
    count3 = Counter(Hives_trace['BCC'])
    true_times3 = count3[1]
    false_times3 = count3[0]
    print(true_times3 / float(false_times3 + true_times3))
    print("Mela", Hives_trace['Mela'])
    count4 = Counter(Hives_trace['Mela'])
    true_times4 = count4[1]
    false_times4 = count4[0]
    print(true_times4 / float(false_times4 + true_times4))
