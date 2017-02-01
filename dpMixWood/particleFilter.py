#Ioannis Papavasileiou
#python implementation of Frank Wood's particle filter for spike sorting
#June 2016

from __future__ import division
from numpy.linalg import *
from scipy.special import gammaln
from scipy import special
import numpy as np
import time
import pdb
import numpy.random, scipy
from decimal import Decimal
import traceback, sys, code
import matplotlib.pyplot as plt
import scipy.io as cio
import sys
import pickle

# This function is written to illustrate online posterior inference
# in an IGMM, of course in a real "online" setting one wouldn't have access
# to all the data upfront.  This code (more generally this technique) is
# factored so that it should be very easy to refactor the code so that the
# data is processed one observation at a time and the particle set is
# also returned having observed each observation

class ParticleFilter(object):
    def __init__(self):
        pass

    def particle_filter_(self,training_data=None,num_particles=None,a_0=None,
        b_0=None,mu_0=None,k_0=None,v_0=None,lambda_0=None,alpha=None,
        partial_labels=None,*args,**kwargs):
        nargin = 10-[training_data,num_particles,a_0,b_0,mu_0,k_0,v_0,lambda_0,alpha,partial_labels].count(None)+len(args)
        pc_max_ind=100000.0
        pc_gammaln_by_2=np.arange(1,pc_max_ind+1)
        pc_gammaln_by_2=special.gammaln(pc_gammaln_by_2 / 2)
        pc_log_pi=np.log(np.pi) 
        pc_log=np.log(np.arange(1,pc_max_ind+1))
        class_id_type='uint8'
        max_class_id=np.iinfo(class_id_type).max

        Y=training_data[:]
        D,T=Y.shape
        N=num_particles
        
        # intialize space for the particles and weights
        particles=np.zeros((N,T,2),dtype=class_id_type)
        weights=np.zeros((N,2))
        #K_plus will be the numebr of classes currently in each particle
        K_plus=np.zeros((N,2),dtype=class_id_type)
        

        # pre-allocate space for per-particle sufficient statistics and other
        # efficiency related variables
        means=np.zeros((D,max_class_id,N,2))
        sum_squares=np.zeros((D,D,max_class_id,N,2))
        inv_cov=np.zeros((D,D,max_class_id,N,2))
        log_det_cov=np.zeros((max_class_id,N,2))
        counts=np.zeros((max_class_id,N,2),dtype=np.uint32)

        #cp is the set of current particles
        #it alternates between 0 and 1
        cp=0

        #not true in our case
        if nargin < 10:
            index_at_which_to_start_estimation=2

            # seat the first customer at the first table in all particles
            particles[:,0,cp]=1
            weights[:,cp]=1. / N

            #initialize the partial sums for seqential covariance and mean updates and
            # precompute the covariance of the posterior predictive student-T with one
            # observation
            y=training_data[:,0]
            yyT=np.dot(y, y.T)
            lp,ldc,ic=lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,1,y,yyT,k_0,mu_0,v_0,lambda_0,nargout=3)
            
            for i in xrange(N):
                means[:,0,i,cp]=y
                sum_squares[:,:,0,i,cp]=yyT
                counts[0,i,cp]=1
                log_det_cov[0,i,cp]=ldc
                inv_cov[:,:,0,i,cp]=ic

        else:
            if partial_labels.shape[1] == 1:
                partial_labels=np.tile(partial_labels.T,(N,1))
            else:
                partial_labels=partial_labels.T
            num_already_seated=partial_labels.shape[1]
            index_at_which_to_start_estimation=num_already_seated
            
            for i in xrange(N):
                particles[i,0:num_already_seated,cp]=partial_labels[i,:]
                weights[:,cp]=1. / N


                num_tables=max(np.unique(partial_labels[i,:]).shape)
                K_plus[:,cp]=num_tables
                
                for k in xrange(num_tables):
                    class_k_inds=partial_labels[i,:] == k
                    Y=training_data[:,class_k_inds]

                    mean_Y=np.mean(Y,axis=1)

                    SS2=np.zeros((Y.shape[0],Y.shape[0]))
                    for z in xrange(np.sum(class_k_inds)):
                        SS2=SS2 + np.dot(Y[:,[z]],Y[:,[z]].T)
                    lp,ldc,ic=self.lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,Y[:,[0]],1,Y[:,[0]],SS2,k_0,mu_0,v_0,lambda_0,nargout=3)
                    

                    means[:,k,i,cp]=mean_Y
                    sum_squares[:,:,k,i,cp]=SS2
                    counts[k,i,cp]=np.sum(class_k_inds)
                    log_det_cov[k,i,cp]=ldc
                    inv_cov[:,:,k,i,cp]=ic

        #init timers
        time_1_obs=0
        total_time=0
        resampleTimes=[]
        sum_K_plus=np.sum(K_plus[:,cp])
        # we need this for memory management later
        
        max_K_plus=np.max(K_plus[:,cp])
        y=training_data[:,[index_at_which_to_start_estimation]]
        E_K_plus=sum_K_plus / N
        
        # M == the number of putatuve particles to generate
        M=sum_K_plus + N
        
        
        putative_particles=np.zeros((2,M))
        putative_weights=np.ones((1,M))
        putative_pf=np.exp(self.lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,0,[],[],k_0,mu_0,v_0,lambda_0)[0])

        ppps=[]
        for t in xrange(index_at_which_to_start_estimation,T):#was T
            tic = time.clock()
            y=training_data[:,[t]]
            
            #si and ei are the starting and ending indexes of the K_plus(n)+1
            #distinct putative particles that are generated from particle n
            si=0
            ei=K_plus[0,cp]+1
            
            sumppps=0
            for n in xrange(N):
                num_options=ei - si # i.e. K_plus(1)+1
                if num_options > max_class_id:
                    raise Exception('K^plus has exceeded the maximum value of '+str(class_id_type))
                
                #copy the old part and choose each table once for the new parts
                putative_particles[0,si:ei]=n
                putative_particles[1,si:ei]=range(num_options)

                #alculate the probability of each new putative particle under the
                #CRP prior alone
                m_k=counts[0:K_plus[n,cp],n,cp].T
                prior=np.concatenate([m_k,np.array([alpha])])
                prior=prior / ((t+1) + alpha)
                
                #update the weights so that the particles (and weights) now represent the
                #predictive distribution
                putative_weights[0,si:ei]=np.dot(weights[n,cp], prior)
                
                #update the weights so that the particles (and weights) now
                #represent the posterior distribution at ''timestep'' t
                posterior_predictive_p=np.zeros((prior).shape)
                for pnp_id in xrange(num_options - 1):
                    lpf=self.lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,counts[pnp_id,n,cp],means[:,[pnp_id],n,cp],
                        sum_squares[:,:,pnp_id,n,cp],k_0,mu_0,v_0,lambda_0,log_det_cov[pnp_id,n,cp],inv_cov[:,:,pnp_id,n,cp])[0]
                    posterior_predictive_p[pnp_id]=np.exp(lpf)
                    
                posterior_predictive_p[-1]=putative_pf
                
                sumppps += posterior_predictive_p.sum()
                
                putative_weights[0,si:ei]=putative_weights[0,si:ei]*(posterior_predictive_p)
                #maintain indexing for putative particle placement in large array
                si=ei
                if n != N-1:
                    ei=si + K_plus[n+1,cp] + 1
            
            ppps.append(sumppps)
            
            ticResample=time.clock()

            # the M weights are computed up to a proportionality so we normalize
            # them here
            putative_weights=putative_weights / np.sum(putative_weights)
            
            c=self.find_optimal_c_(putative_weights,N)
            # find pass-though ratio
            pass_inds=(putative_weights > 1. / c).ravel()
            not_pass_ids = (putative_weights <= 1. / c).ravel()
            num_pass=np.sum(pass_inds)
            if cp == 0:
                np_var=1
            else:
                np_var=0   # you didn't extract the number of rows from the matrix and hardcoded 7.  If I test the code with another matrix there could be more rows.

            yyT=y.dot(y.T)

            if num_pass > 0:
                particles[0:num_pass,0:t ,np_var]=particles[[putative_particles[0,pass_inds]],0:t ,cp]
                particles[0:num_pass,t,np_var]=putative_particles[1,pass_inds]
                weights[0:num_pass,np_var]=putative_weights[0,pass_inds]
                
                passing_class_id_ys=putative_particles[1,pass_inds]
                passing_orig_partical_ids=putative_particles[0,pass_inds]
                for npind in xrange(num_pass):
                    class_id_y=passing_class_id_ys[npind]
                    originating_particle_id=passing_orig_partical_ids[npind]
                    originating_particle_K_plus=K_plus[originating_particle_id,cp]
                    
                    new_count=counts[class_id_y,originating_particle_id,cp] + 1
                    
                    if new_count == 1:
                        K_plus[npind,np_var]=originating_particle_K_plus + 1
                    else:
                        K_plus[npind,np_var]=originating_particle_K_plus
                    
                    counts[0:max_K_plus,npind,np_var]=counts[0:max_K_plus,originating_particle_id,cp]
                    counts[class_id_y,npind,np_var]=new_count
                    old_mean=means[:,class_id_y,originating_particle_id,cp]
                    
                    means[:,0:max_K_plus,npind,np_var]=means[:,0:max_K_plus,originating_particle_id,cp]
                    means[:,class_id_y,npind,np_var]=old_mean + (1. / new_count) * (y[:,0] - old_mean)
                    
                    sum_squares[:,:,0:max_K_plus,npind,np_var]=sum_squares[:,:,0:max_K_plus,originating_particle_id,cp]
                    sum_squares[:,:,class_id_y,npind,np_var]=sum_squares[:,:,class_id_y,originating_particle_id,cp] + yyT
                    
                    
                    
                    # here we use a hidden feature of  lp_tpp_helper  in that
                    # it will calculate the log_det_cov and the inv_cov for us
                    # automatically.  we don't care about lp here at all
                    lp,ldc,ic=self.lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,new_count,means[:,[class_id_y],npind,np_var],sum_squares[:,:,class_id_y,npind,np_var],k_0,mu_0,v_0,lambda_0,nargout=3)
                    #lp hellper 12 arguments
                    
                    log_det_cov[0:max_K_plus,npind,np_var]=log_det_cov[0:max_K_plus,originating_particle_id,cp]
                    log_det_cov[class_id_y,npind,np_var]=ldc
                    inv_cov[:,:,0:max_K_plus,npind,np_var]=inv_cov[:,:,0:max_K_plus,originating_particle_id,cp]
                    inv_cov[:,:,class_id_y,npind,np_var]=ic
                    
            if N - num_pass > 0:
                weights[num_pass-1:,np_var]=1 / c
                picked_putative_particles=self.stratified_resample_(putative_particles[:,not_pass_ids],putative_weights[0,not_pass_ids],N - num_pass)
                #pdb.set_trace()
                npind=num_pass
                for ppind in xrange(N - num_pass):
                    class_id_y=picked_putative_particles[1,ppind]
                    originating_particle_id=picked_putative_particles[0,ppind]
                    originating_particle_K_plus=K_plus[originating_particle_id,cp]
                    
                    particles[npind,0:t,np_var]= np.append(particles[originating_particle_id,0:t - 1,cp],np.array(class_id_y))
                    
                    new_count=counts[class_id_y,originating_particle_id,cp] + 1
                    K_plus[npind,np_var]=originating_particle_K_plus
                    if new_count == 1:
                        K_plus[npind,np_var]=originating_particle_K_plus + 1
                    

                    counts[0:max_K_plus,npind,np_var]=counts[0:max_K_plus,originating_particle_id,cp]
                    counts[class_id_y,npind,np_var]=new_count
                    
                    old_mean=means[:,class_id_y,originating_particle_id,cp]
                    
                    means[:,0:max_K_plus,npind,np_var]=means[:,0:max_K_plus,originating_particle_id,cp]
                    means[:,class_id_y,npind,np_var]=old_mean + np.dot((1. / new_count), (y[:,0] - old_mean))
                    sum_squares[:,:,0:max_K_plus,npind,np_var]=sum_squares[:,:,0:max_K_plus,originating_particle_id,cp]
                    sum_squares[:,:,class_id_y,npind,np_var]=sum_squares[:,:,class_id_y,originating_particle_id,cp] + yyT
                    
                    # here we use a hidden feature of  lp_tpp_helper  in that
                    # it will calculate the log_det_cov and the inv_cov for us
                    # automatically.  we don't care about lp here at all
                    lp,ldc,ic=self.lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,new_count,means[:,[class_id_y],npind,np_var],sum_squares[:,:,class_id_y,npind,np_var],k_0,mu_0,v_0,lambda_0,nargout=3)
                    #lp helper 12 arguments
                    log_det_cov[0:max_K_plus,npind,np_var]=log_det_cov[0:max_K_plus,originating_particle_id,cp]
                    log_det_cov[class_id_y,npind,np_var]=ldc
                    inv_cov[:,:,0:max_K_plus,npind,np_var]=inv_cov[:,:,0:max_K_plus,originating_particle_id,cp]

                    inv_cov[:,:,class_id_y,npind,np_var]=ic
                    
                    npind=npind + 1
            cp=np_var
            time_1_obs=time.clock() - tic
            
            
            sum_K_plus=np.sum(K_plus[:,cp])
            # we need this for memory management later
            
            max_K_plus=np.max(K_plus[:,cp])
            E_K_plus=sum_K_plus / N
            total_time=total_time + time_1_obs
            
            if t == 2:
                print 'CRP PF:: Obs: ',t,'/',T
            else:
                if t % 5 == 0:
                    rem_time=(time_1_obs * 0.05 + 0.95 * (total_time / t)) * T - total_time
                    if rem_time < 0:
                        rem_time=0
                    print 'CRP PF:: Obs: ',t,'/',T,', Rem. Time: ',self.secs2hmsstr(rem_time),', Ave. Time: ',self.secs2hmsstr((total_time / (t - 2))),', Elaps. Time: ',self.secs2hmsstr(total_time),', E[K^+] ',E_K_plus
            # M == the number of putatuve particles to generate
            M=sum_K_plus + N
            
            putative_particles=np.zeros((2,M))
            putative_weights=np.ones((1,M))
            
            putative_pf=np.exp(self.lp_tpp_helper_(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,0,[],[],k_0,mu_0,v_0,lambda_0)[0])
            
            resampleTimes.append(time.clock()-ticResample)
            
        ret_particles=np.squeeze(particles[:,:,cp])
        ret_weights=np.squeeze(weights[:,cp])
        ret_K_plus=np.squeeze(K_plus[:,cp])
        return ret_particles,ret_weights,ret_K_plus
    #__end__ particle_fileter

    

    def stratified_resample_(self,x=None,w=None,N=None,*args,**kwargs):
        D,M=x.shape
        ni=np.random.permutation(M)
        x=x[:,ni]
        w=w[ni]
        rx=np.zeros((D,N))
        rw=np.zeros((1,N))
        cdf=np.cumsum(w,axis=0)

        cdf[-1]=1
        
        randomNumber = np.random.rand()
        p=np.linspace(randomNumber * (1. / N),1,num=N)
        picked=np.zeros((1,M))
        j=0
        for i in xrange(N):
            while j < M and  cdf[j] < p[i]:
                j=j + 1
            picked[0,j]=picked[0,j] + 1
        rind=0
        for i in xrange(M):
            if (picked[0,i] > 0):
                for j in xrange(Decimal(picked[0,i])):
                    rx[:,rind]=x[:,i]
                    rw[0,rind]=w[i]
                    rind=rind + 1
        rw=rw / np.sum(rw)
        return rx

    def find_optimal_c_(self,Q=None,N=None,*args,**kwargs):
        Q=np.sort(Q)[0][::-1]
        c=0
        k=0
        M=max(Q.shape)
        k_old= -np.inf
        while k_old != k:

            k_old=k
            c=(N - k) / np.sum(Q[k:])
            k=k + np.sum(Q[k:M] * c > 1)
        return c

    def lp_tpp_helper_(self,pc_max_ind=None,pc_gammaln_by_2=None,pc_log_pi=None,
            pc_log=None,y=None,n=None,m_Y=None,SS=None,k_0=None,mu_0=None,
            v_0=None,lambda_0=None,log_det_Sigma=None,inv_Sigma=None,*args,**kwargs):
        nargin = 14-[pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,n,m_Y,SS,k_0,mu_0,v_0,lambda_0,log_det_Sigma,inv_Sigma].count(None)+len(args)        
        d=len(y)

        if n != 0:
            mu=np.dot(k_0 / (k_0 + n), mu_0) + np.dot(n / (k_0 + n),m_Y)
            v_n=v_0 + n
        else:
            mu=mu_0
            v_n=v_0
        
        v=v_n - 2 + 1
        if nargin < 13:
            if n != 0:
                k_n=k_0 + n
                S=(SS - np.dot(np.dot(n, m_Y), m_Y.T))
                zm_Y=m_Y - mu_0
                lambda_n=lambda_0 + S + np.dot(np.dot(k_0 * n / (k_0 + n), zm_Y), zm_Y.T)
            else:
                k_n=k_0
                lambda_n=lambda_0
            Sigma=np.dot(lambda_n, (k_n + 1) / (k_n * (v_n - 2 + 1)))
            log_det_Sigma=np.log(det(Sigma))
            inv_Sigma=inv(Sigma)
        vd=v + d
        
        if vd < pc_max_ind:
            d2=d / 2.
            lp=pc_gammaln_by_2[vd-1] - (pc_gammaln_by_2[v-1] + d2 * pc_log[v-1] + d2 * pc_log_pi) - 0.5 * log_det_Sigma - (vd / 2.) * np.log(1 + (1. / v) * np.dot(np.dot((y - mu).T,inv_Sigma),y - mu))
        else:
            lp=gammaln((v + d) / 2) - (gammaln(v / 2) + (d / 2) * log_(v) + (d / 2) * pc_log_pi) - 0.5 * log_det_Sigma - ((v + d) / 2) * np.log(1 + (1 / v) * np.dot(np.dot((y - mu).T, inv_Sigma), (y - mu))) 
        return lp,log_det_Sigma,inv_Sigma



    def secs2hmsstr(self,secs=None,*args,**kwargs):
        days=np.floor(secs / (3600 * 24))
        rem=(secs - (days * 3600 * 24))
        hours=np.floor(rem / 3600)
        rem=rem - (hours * 3600)
        minutes=np.floor(rem / 60)
        rem=rem - minutes * 60
        secs=rem
        if 0 == days:
            _str=str(hours)+':'+'%02d'%minutes+':'+'%02.2f'%secs
        else:
            if 1 == days:
                _str='1 Day + '+str(hours)+':'+'%02d'%minutes+':'+'%02.2f'%secs
            else:
                _str=str(days)+' Days + '+str(hours)+':'+'%02d'%minutes+':'+'%02.2f'%secs
        return _str

    
    def save(self,var,name):
        d={};
        d[name]=var
        cio.savemat('test.mat',d)

'''
#test the code here
#load particle filter data
partData = scipy.io.loadmat('particleFilterData.mat')
#load dictionary in workspace
v_0=5
#SSleft is a matrix where observations are in rows and features in columns
SSleft=partData['SSleft']
num_particles=partData['num_particles'][0,0]
a_0=partData['a_0']
b_0=partData['b_0']
mu_0=partData['mu_0']
k_0=partData['k_0'][0,0]
lambda_0=partData['lambda_0']
alpha_record=partData['alpha_record']
class_id_samples=partData['class_id_samples']

particleFilter = ParticleFilter()
spike_sortings,spike_sorting_weights,number_of_neurons_in_each_sorting=particleFilter.particle_filter_(SSleft.T,num_particles,a_0,b_0,mu_0,k_0,v_0,lambda_0,np.mean(alpha_record),class_id_samples-1)
cio.savemat('particleResults.mat',{'spike_sortings':spike_sortings,'spike_sorting_weights':spike_sorting_weights,'number_of_neurons_in_each_sorting':number_of_neurons_in_each_sorting})
'''
