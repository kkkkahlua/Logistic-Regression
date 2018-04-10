import numpy as np
from math import exp
from math import log

class Logistic:
    def __init__(self,samples,labels):
        self.n_samples=len(samples)
        self.n_attrs=samples.shape[1]  # including 1
        self.samples=samples
        self.labels=labels

    def get_p1(self,beta,index):
        v2 = np.zeros((1, self.n_attrs))
        v2[0] = self.samples[index]
        tmp = exp(np.dot(v2, beta))
        return tmp / (1 + tmp)

    def get_p0(self,beta,index):
        return 1-self.get_p1(beta,index)

    def get_grad(self,beta):
        grad=np.zeros((self.n_attrs,1))
        for i in range(self.n_samples):
            fact = self.get_p1(beta,i)-self.labels[i][0]
            print(fact.shape)
            grad+=fact*self.samples[i].T
        return grad

    def get_Hessian(self,beta):
        Hessian=np.zeros((self.n_attrs,self.n_attrs))
        for i in range(self.n_samples):
            p1=self.get_p1(beta,i)
            Hessian+=(np.dot(self.samples[i].T,self.samples[i])*p1*(1-p1))
        return Hessian

    def log_likelihood_func(self,beta):
        ret=0
        for i in range(self.n_samples):
            ret+=(-self.labels[i]*np.dot(beta.T,self.samples[i].T)+log(1+exp(np.dot(beta.T,self.samples[i].T))))
        return ret

    def get_alpha(self,beta,r,grad):
        delta = 0.5
        sigma = 0.25
        m=0
        while True:
            tmp=delta**m
            if self.log_likelihood_func(beta+tmp*r)<=\
                   self.log_likelihood_func(beta)+np.dot(sigma*tmp,np.dot(grad.T,r)):
                return delta**m
            else:
                m += 1

    def Newton(self,maxturn=10000,eps=1e-5):
        turn=0
        beta=np.zeros((self.n_attrs,1))
        prenorm=float('inf')
        while True:
            grad=self.get_grad(beta)
            gradnorm=np.linalg.norm(grad)
            print("gradnorm=",gradnorm," turn=",turn)
            if turn>=maxturn or gradnorm<=eps or \
                    (prenorm-gradnorm<=eps and prenorm-gradnorm>=0):
                break
            prenorm=gradnorm
            Hessian=self.get_Hessian(beta)
            #print("Hessian=\n",Hessian)
            Hessian_inverse=np.linalg.inv(Hessian)
            r=-np.dot(Hessian_inverse,grad)
            alpha=self.get_alpha(beta,r,grad)
            beta=beta+alpha*r
            turn=turn+1

        if turn>=maxturn:
            print("out of iteration limit")
        return beta

def main():
    samples = np.mat([[1, 2, 3,1], [4, 5, 6,1], [2, 3, 1,1],[7,8,9,111]])
    print(samples[0].shape)
    print(samples[0])
    labels=[1,1,1,1,1]
    c=Logistic(samples,labels)
    a=c.Newton()
    print(a)
if __name__ == '__main__':
    main()