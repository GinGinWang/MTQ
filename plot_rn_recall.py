import matplotlib.pyplot as plt

x = [0.1*(i+1) for i in range(9)]

# average
tn_over_n = [1,1,1,0.99913,0.99913,0.99913,0.99786,0.99614,0.99203]
recall = [1,1,1,0.99968,0.99979,0.99986,0.99977,0.99976,0.99979]

plt.figure(0)
plt.plot(x, tn_over_n,'s-',color = 'r',label="TN/N")#
plt.plot(x, recall,'o-',color = 'g',label="Recall")#o-

plt.xlabel("novel-ratio")
plt.ylabel("metric")

plt.legend(loc = "upper right")
plt.savefig("distgraph/TN_Recall_mnist_average.png")

# class 8 -test-set(all)

tn_over_n = [1,1,1,1,1,1,1,1,1]
recall = [1,1,1,1,1,1,1,1,1]

plt.figure(1)
plt.plot(x, tn_over_n,'s-',color = 'r',label="TN/N")#
plt.plot(x, recall,'o-',color = 'g',label="Recall")#o-

plt.xlabel("novel-ratio")
plt.ylabel("metric")

plt.legend(loc = "upper right")
plt.savefig("distgraph/TN_Recall_mnist_class8.png")

# class 8 -test_set(3,8)


# tn_over_n = []
# recall = []

# plt.figure(2)
# plt.plot(x, tn_over_n,'s-',color = 'r',label="TN/N")#
# plt.plot(x, recall,'o-',color = 'g',label="Recall")#o-

# plt.xlabel("novel-ratio")
# plt.ylabel("metric")

# plt.legend(loc = "upper right")
# plt.savefig("TN_Recall_mnist_class8_test_3_8.png")

# # class 8 - testset_set(1,8)


tn_over_n = [1,1,1,1,0.99510,0.99510,0.99510,0.99912]
recall = [1,1,1,1,0.99897,0.99912,0.99912,0.99912]

plt.figure(3)
plt.plot(x, tn_over_n,'s-',color = 'r',label="TN/N")#
plt.plot(x, recall,'o-',color = 'g',label="Recall")#o-

plt.xlabel("novel-ratio")
plt.ylabel("metric")

plt.legend(loc = "upper right")
plt.savefig("TN_Recall_mnist_class8_test_1_8.png")


