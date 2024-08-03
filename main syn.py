import numpy as np
from  scipy.stats import rayleigh
import matplotlib.pyplot as plt
#-------------------------------------------
from sac_torch import Agent
# np.random.seed(1373)
#%%
x_lim=100
y_lim=100
number_HD=40
number_AP=4
number_subchannel=32
#$-------------------
beta_max=.95
beta_min=.05
#$-------------------
scale_power=1
E_max=(.01)*scale_power
P_max_AP=(0.1)*scale_power
sigma_noise=10e-17
bandwith=200e3


N_number_types=5
Ttype=10
f_d=0.9

#%Delay 
D_th=60 # milisecond

#%%
##########################################################################################
class location_channel:
    def __init__(self):
        self.loc_AP=np.random.rand(2,number_AP)*x_lim
        self.accosiator_HD_AP=np.zeros([number_HD,number_AP])
        #------------------------------------------------------------------
        # self.loc_AP[:,0] = x_lim / 3
        # self.loc_AP[:,3] = 2 * x_lim / 3
        # self.loc_AP[0, 1] = 2 * x_lim / 3
        # self.loc_AP[1, 1] = x_lim / 3
        # self.loc_AP[1, 2] = x_lim / 3
        # self.loc_AP[1,2] = 2 * x_lim / 3
        #------------------------------------------------------------------
        self.loc_HD=np.random.rand(2,number_HD)*x_lim
        #------------------------------------------------------------------
        self.H=np.zeros([number_AP,number_subchannel,number_HD])
        self.H_hat=np.zeros([number_AP,number_subchannel,number_HD])
        #------------------------------------------------------------------
        
    def _(self):
        for u in range(number_HD):
          self.mem_ap=[]
          self.xu = self.loc_HD[0,u]
          self.yu = self.loc_HD[1,u]
          for ap in range(number_AP):
              dd=(self.loc_AP[0,ap]-self.xu)**2 + (self.loc_AP[1,ap]-self.yu)**2
              dd=dd**.5
              self.mem_ap.append(dd)
              dd=dd**(-3)
              H_u=rayleigh.rvs(np.ones(number_subchannel))*dd
              H_hat_u=rayleigh.rvs(np.ones(number_subchannel))*dd
              self.H[ap,:,u]=H_u
              self.H_hat[ap,:,u]=H_hat_u
              
          self.mem_ap=np.array(self.mem_ap)
          temp_ap=np.argmin(self.mem_ap)
          self.accosiator_HD_AP[u,temp_ap]=1
          
        return self.H, self.H_hat,self.accosiator_HD_AP
#%%##########################################################################################
class val_gen:
    def __init__(self, mat_rate_up, mat_rate_down, mat_types):
        self.mat_rate_up=mat_rate_up
        self.mat_rate_down=mat_rate_down
        self.mat_types=mat_types

    def _m_(self,t):
        self.mat_types[t,:]=np.random.randint(0,N_number_types,number_HD)
        self.X=np.zeros([number_HD])
        if t>Ttype:
            V_ini=0
            V_th=1e6
            self.done_=0
            for u in range(number_HD):
                self.X[u]=len(set(self.mat_types[t-Ttype:t,u]))
                self.inv_X=1/self.X[u]
                self.v_u=self.inv_X*(self.mat_rate_up[u]+self.mat_rate_down[u])
                V_ini+=self.v_u
                
            if V_ini>V_th:
                self.done_=1
        else:
            self.done_=1
        return self.done_
#%%##########################################################################################
class delay_cal:
    def __init__(self,mat_rate_up_per_u, mat_rate_down_per_u, beta):
        self.mat_rate_up_per_u=mat_rate_up_per_u
        self.mat_rate_down_per_u=mat_rate_down_per_u
        self.beta=beta
        self.C_u= 1e6  # For HCD   # CPU cycle
        

        f_d=.9
        self.eta_proc=.01
        self.C_b=abs(np.sqrt(sigma_C)*np.random.randn(number_AP)+np.sqrt(sigma_C*2/np.pi))
        self.J=abs(np.sqrt(sigma_T)*np.random.randn(number_HD)+np.sqrt(sigma_T*2/np.pi))
        
    def __processing(self):
        self.mat_delay_Proc_u=np.zeros([number_HD])
        for u in range(number_HD):
            self.mat_delay_Proc_u[u]= (((1-self.beta[u])*self.J[u]/self.C_u)+(f_d*self.beta[u]*self.J[u]/self.C_b[0]))*self.eta_proc
        
        return  self.mat_delay_Proc_u
        
    def __transmission(self):
        self.mat_delay_Trans_u=np.zeros([number_HD])
        for u in range(number_HD):
            self.mat_delay_Trans_u[u]= self.beta[u]*(self.J[u]*(1/self.mat_rate_up_per_u[u])+f_d*self.J[u]*(1/self.mat_rate_down_per_u[u]))
       
        return  self.mat_delay_Trans_u
        
    def __que(self):
        self.mat_delay_que_u=np.zeros([number_HD])
        return  self.mat_delay_que_u
    
    def _total_(self):
        self.mat_done=np.zeros([number_HD])
        self.mat_Trans_u=delay_cal.__transmission(self)
        self.mat_Que_u=delay_cal.__que(self)
        self.mat_Proc_u=delay_cal.__processing(self)
        for u in range(number_HD):
            if ((self.mat_Trans_u[u]+self.mat_Proc_u[u]+self.mat_Que_u[u]))<D_th:
                self.mat_done[u]=1
        return self.mat_done,   
#%%##########################################################################################

class rate_cal:
    def __init__(self,rho,rho_hat,p,p_hat,H,H_hat):
        self.rho=rho
        self.rho_hat=rho_hat
        self.p=p
        self.p_hat=p_hat
        self.H=H
        self.H_hat=H_hat
        self.min_rate_down=min_rate_down
        self.min_rate_up=min_rate_up

    def __downlink__(self):
        self.done_total_down = 0
        self.mat_done_down=np.zeros([number_HD])
        self.mat_rate_down_per_u=np.zeros([number_HD])
        self.mat_rate_down = np.zeros([number_AP,number_subchannel,number_HD])
        for u in range(number_HD):
            self.comul=0
            for ap in range(number_AP):
                for k in range(number_subchannel):
                    if self.rho[ap,k,u]>0:
                        ph = self.p[ap,k,u]*self.H[ap,k,u]*self.rho[ap,k,u]
                        I_intr=rate_cal.__inter__(self, ap, k, u, self.H, self.p, self.rho)
                        SINR = ph/(sigma_noise+I_intr) 
                        if SINR>0:
                            self.mat_rate_down[ap,k,u]=bandwith*np.log2(1+SINR)
                            self.comul+=bandwith*np.log2(1+SINR)
                            
            self.mat_rate_down_per_u[u]=self.comul                        
            if self.comul>=self.min_rate_down: #%#  To be in Mbyte
                self.mat_done_down[u]=1
        if np.sum(self.mat_done_down)==number_HD:
            self.done_total_down=1
            
        return self.mat_rate_down, self.done_total_down, self.mat_done_down, self.mat_rate_down_per_u
#---------------------------------------------------------
    def __uplink__(self):
        self.done_total_up = 0
        self.mat_done_up = np.zeros([number_HD])    
        self.mat_rate_up=np.zeros([number_AP,number_subchannel,number_HD])
        self.mat_rate_up_per_u=np.zeros([number_HD])    
        for u in range(number_HD):
            self.comul=0
            for ap in range(number_AP):
                for k in range(number_subchannel):
                    if self.rho_hat[ap,k,u]>0:
                        ph = self.p_hat[ap,k,u]*self.H_hat[ap,k,u]
                        I_intr=rate_cal.__inter__(self, ap, k, u, self.H_hat, self.p_hat, self.rho_hat)
                        SINR = ph/(sigma_noise+I_intr) 
                        if SINR>0:
                            self.mat_rate_up[ap,k,u]=bandwith*np.log2(1+SINR)
                            self.comul+=bandwith*np.log2(1+SINR)

            self.mat_rate_up_per_u[u]=self.comul
            if self.comul>=self.min_rate_up:
                self.mat_done_up[u]=1
        if np.sum(self.mat_done_up)==number_HD:
            self.done_total_up=1
        return self.mat_rate_up, self.done_total_up, self.mat_done_up, self.mat_rate_up_per_u
    
    def __inter__(self,ap,k,u,H,p,rho):
        self.I_intr=0
        for uu in range(number_HD):
            if uu != u:
                for app in range(number_AP):
                    if app != ap:
                        self.I_intr+=H[app,k,uu]*p[app,k,uu]*rho[app,k,uu]
        return  self.I_intr
#%%---------------------------------------------------------------------
class state_cal:
    def __init__(self,H,H_hat):
        self.H=H
        self.H_hat=H_hat
        self.state_size = 2*self.H.size 
        self.state=np.zeros(self.state_size)
        
    def _(self):
        self.H_resh=np.reshape(self.H,[1,self.H.size])
        self.H_hat_resh=np.reshape(self.H_hat,[1,self.H_hat.size])
        self.state[0:self.H.size] = self.H_resh
        self.state[self.H.size:2*self.H.size] = self.H_hat_resh
        self.state=self.state/max(self.state)
        return self.state
#%%---------------------------------------------------------------------
class __run__:
    def __init__(self,T):
        self.T=T
        self.mat_Reward=np.zeros([self.T])
        self.mat_rate=np.zeros([self.T])
        self.mat_Val_time=np.zeros([self.T,number_HD])
        self.mat_power_consumed=np.zeros([self.T])
        alpha=.00001
        beta=.000001
        self.n_actions=5*number_AP*number_subchannel*number_HD + number_HD
        input_dims=2*number_AP*number_subchannel*number_HD 
        self.agent=Agent(alpha, beta, self.n_actions, input_dims)
        self.mat_energy=E_max*np.ones(number_HD)
        self.var=1
        self.decay_var=.0
        self.LC=location_channel()
        self.ep_rewardall=[]
        self.ep_rate=[]
        self.mem_critc=[]
        self.mem_act=[]
        self.mat_types=np.zeros([T,number_HD])

    def __(self):
        for t in range(self.T):
            print(t)
            w=100
            if t>0:
                if np.mod(t,w)==0:
                    aaa=len(self.ep_rewardall)
                    mean_ep_rewardall=[]
                    for i in range(aaa-w) :
                        temp_value= np.sum(self.ep_rate[i:-aaa + w+i])/w
                        mean_ep_rewardall.append(temp_value) 
                    plt.plot(mean_ep_rewardall, label='Mean of Rate')
                    plt.xlabel("Episode")
                    plt.ylabel("Mean of Rate")
                    plt.legend()
                    plt.show()    
                    #------------------------------------------
                    # mean_ep_c=[]
                    # aaa=len(self.mem_critc)
                    # mean_ep_rewardall=[]
                    # for i in range(aaa-w) :
                    #     temp_value= np.sum(self.mem_critc[i:-aaa + w+i])/w
                    #     mean_ep_rewardall.append(temp_value) 
                    # plt.plot(mean_ep_rewardall, label='Mean of Loss')
                    # plt.xlabel("Episode")
                    # plt.ylabel("Mean of loss_crit")
                    # plt.legend()
                    # plt.show()    
                    # #--------------------------------------
                    # mean_ep_a=[]
                    # aaa=len(self.mem_act)
                    # mean_ep_rewardall=[]
                    # for i in range(aaa-w) :
                    #     temp_value= np.sum(self.mem_act[i:-aaa + w+i])/w
                    #     mean_ep_rewardall.append(temp_value) 
                    # plt.plot(mean_ep_rewardall, label='Mean of Loss')
                    # plt.xlabel("Episode")
                    # plt.ylabel("Mean of loss_act")
                    # plt.legend()
                    # plt.show()  
                    
                    
            self.reward=-50
            self.H, self.H_hat, self.accosiator_HD_AP = self.LC._()
            SC=state_cal(self.H, self.H_hat)
            self.state=SC._()
            #-----------------------------------------
            self.action = self.agent.choose_action(self.state)
            #--------------------------
            self.action_pure = np.copy(self.action)
            #--------------------------
            self.var=self.var*self.decay_var
            self.noise=np.random.randn(self.n_actions)
            # self.noise=self.noise*self.var          
            self.action= self.action + self.noise
            self.action = np.clip(self.action,-1,1)
            #-----------------------------------------
            MA=mapping(self.action, self.accosiator_HD_AP)
            self.rho, done_u1 = MA._rho_down()
            self.rho_hat, done_u2=MA._rho_up()
            self.p, done_u3=MA._power_down(self.rho)
            self.p_hat, done_u4=MA._power_up(self.rho_hat)
            self.beta=MA._beta()
            #--------------------------------------------
            if done_u1==1:
                if done_u2==1:
                    if done_u3==1:
                        if done_u4==1:
                            print("done_prime")
            #--------------------------------------
            RC=rate_cal(self.rho, self.rho_hat, self.p, self.p_hat, self.H, self.H_hat)
            self.mat_rate_down, self.done_total_down, self.mat_done_down, self.mat_rate_down_per_u = RC.__downlink__()
            self.mat_rate_up, self.done_total_up, self.mat_done_up, self.mat_rate_up_per_u = RC.__uplink__()
            #==================================================================
            self.VG_=val_gen(self.mat_rate_up_per_u, self.mat_rate_down_per_u,self.mat_types)
            self.__done_Value_=self.VG_._m_(t)
            #==================================================================
            self.DC=delay_cal(self.mat_rate_up_per_u, self.mat_rate_down_per_u, self.beta)
            self.mat_done =self.DC._total_()
            self.done_delay=0
            if np.sum(self.mat_done)==number_HD:
                self.done_delay=1
            #==================================================================
            
            if self.done_total_down==1:
                if self.done_total_up==1:
                    print("done_second")
                    self.flag_have=np.zeros([number_HD])
                    self.flag_gen=np.zeros([number_HD])
                    self.mat_power_consumed[t]=(np.sum(self.p) + np.sum(self.p_hat))
                    self.reward =  (np.sum(self.mat_rate_down)+np.sum(self.mat_rate_up))/(1e6)  #10-1*((np.sum(self.p) + np.sum(self.p_hat)))
                    self.ep_rate.append((np.sum(self.mat_rate_down)+np.sum(self.mat_rate_up)))
                    self.mat_rate[t]=(np.sum(self.mat_rate_down)+np.sum(self.mat_rate_up))
                else:
                    self.reward -= 100
                    self.ep_rate.append(0)
                    
            # print(self.mat_energy)
            # print("reward:::",self.reward)
            # print(self.action)
            self.flag_gen=np.zeros([number_HD])
            SC=state_cal(self.H, self.H_hat)
            self.state_new=SC._()
            
            self.agent.memorize(self.state, self.action_pure, self.reward, self.state_new)
            self.agent.replay()
            # self.mem_critc.append(loss_c)
            # self.mem_act.append(loss_a)
            #--------------------------------------
            self.mat_Reward[t] = self.reward
            self.ep_rewardall.append(self.reward)
            
        return  self.mat_Reward,  self.mat_Val_time, self.mat_power_consumed,  self.mat_rate
#%%---------------------------------------------------------------
T=4200
sigma_C=100e6    # For MEC servers
sigma_T=.1e6   # For Tasks
RUN=__run__(T)
mat_Reward,  mat_Val_time, mat_power_consumed, mat_rate =RUN.__()
 
                    
                

