import scipy.io as sio
class Theta:
    w1=0.1
    w2=0.1
    w3=0.1
    b=0.1
    pass

def linear_regress(y,X):
    # 将三维的数组按列切分成三个n维向量
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    theta = Theta()
    beta = 0.0001 # 阈值
    rate = 0.05 # 学习率
    times = 0
    experience_risk_pre = 0
    while(True):
        theta_w1=0
        theta_w2=0
        theta_w3=0
        b=0
        # 求各个参数的偏导数,梯度下降计算各个参数
        for i in range(len(y)):
            # 计算偏导数
            theta_w1 +=2 * x1[i]*( (theta.w1*x1[i]) + (theta.w2*x2[i]) + (theta.w3*x3[i]) + theta.b - y[i] )/len(y)
            theta_w2 +=2 * x2[i]*( (theta.w1*x1[i]) + (theta.w2*x2[i]) + (theta.w3*x3[i]) + theta.b - y[i] )/len(y)
            theta_w3 +=2 * x3[i]*( (theta.w1*x1[i]) + (theta.w2*x2[i]) + (theta.w3*x3[i]) + theta.b - y[i] )/len(y)
            b += 2 * ( (theta.w1*x1[i]) + (theta.w2*x2[i]) + (theta.w3*x3[i]) + theta.b - y[i] )/len(y)
        # 梯度下降更新参数
        theta.w1 = theta.w1 - rate*theta_w1
        theta.w2 = theta.w2 - rate*theta_w2
        theta.w3 = theta.w3 - rate*theta_w3
        theta.b = theta.b - rate*b
        # 计算此时的经验风险
        experience_risk = 0;
        for i in range(len(y)):
            experience_risk+=( (theta.w1*x1[i]) + (theta.w2*x2[i]) + (theta.w3*x3[i]) + theta.b - y[i] )**2/len(y)
        #print("第{}次循环,风险{}".format(times,experience_risk))
        if((experience_risk - experience_risk_pre)**2<=beta):
            break
        experience_risk_pre = experience_risk
        times+=1
    return theta
def y_hat(theta,x_test):
    x1 = x_test[:, 0]
    x2 = x_test[:, 1]
    x3 = x_test[:, 2]
    y=[]
    for i in range(len(x_test)):
        y.append((theta.w1*x1[i]) + (theta.w2*x2[i]) + (theta.w3*x3[i]) + theta.b)
    return y


if __name__ == "__main__":
    p2 = sio.loadmat('data/p2.mat')
    # x_in
    x_in = p2['X_in']
    y_noisy = p2['y_noisy']
    # theta
    theta = linear_regress(y_noisy,x_in)
    print("w1:{} w2:{} w3:{} b{}".format(theta.w1,theta.w2,theta.w3,theta.b))
    # Y_hat
    Y_hat = y_hat(theta,x_in)
    y_true = p2['y_true']
    # variance
    variance = 0
    for i in range(len(y_true)):
        variance += (y_true[i] - Y_hat[i])**2/len(y_true)
    print("方差{}".format(variance))

