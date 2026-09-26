import numpy as np
sig=5.67e-8
def climate(kind):
    h=np.arange(24)+0.5
    if kind=='hotdry':   Tmin,Tmax,Gpk,esky=29,43,1000,0.74
    else:                Tmin,Tmax,Gpk,esky=26,33,900,0.86
    Ta=(Tmin+Tmax)/2+(Tmax-Tmin)/2*np.cos(2*np.pi*(h-15)/24)
    G=np.clip(Gpk*np.sin(np.pi*(h-6)/13),0,None)  # sunrise 6, sunset 19
    return h,Ta,G,esky
def wall_G(h):
    # west wall: diffuse morning, beam afternoon peak ~750 at 16:30
    Gd=np.clip(120*np.sin(np.pi*(h-6)/13),0,None)
    Gb=np.clip(700*np.sin(np.pi*(h-12.5)/6.5),0,None)*(h>12.5)
    return Gd+Gb
def surf(alpha,eps,G,Ta,Tsky_eff4,hc,U,Ti,Fsky=1.0,Tgnd=None):
    # solve alpha G = hc(Ts-Ta)+eps sig (Ts^4 - Tenv^4) + U'(Ts-Ti), U' = conductance from outer surface to inside air
    Ts=Ta.copy()+10
    Tenv4=Fsky*Tsky_eff4+(1-Fsky)*((Tgnd if Tgnd is not None else Ta)+273.15)**4
    for _ in range(60):
        f=alpha*G-hc*(Ts-Ta)-eps*sig*((Ts+273.15)**4-Tenv4)-U*(Ts-Ti)
        df=-hc-4*eps*sig*(Ts+273.15)**3-U
        Ts=Ts-f/df
    return Ts
def daily_gain(SR,eps,U,kind='hotdry',element='roof',Ti=24.0,hc=None):
    h,Ta,G,esky=climate(kind)
    Tsky4=esky*(Ta+273.15)**4
    Rsi=0.10 if element=='roof' else 0.13
    Uprime=1/(1/U-Rsi-0.04)  # surface-to-inside conductance (remove Rse,Rsi from nominal U)
    if element=='roof':
        Ts=surf(1-SR,eps,G,Ta,Tsky4,hc or 12,Uprime,Ti)
    else:
        Gw=wall_G(h)
        Ts=surf(1-SR,eps,Gw,Ta,Tsky4,hc or 8,Uprime,Ti,Fsky=0.5,Tgnd=Ta+4)
    q=Uprime*(Ts-Ti)   # W/m2 into interior (quasi-steady; 24h sum ~valid for massive too)
    return q.sum()/1000, q.max(), Ts.max()  # kWh/m2/day
def netcool(SR,eps,kind):
    h,Ta,G,esky=climate(kind)
    T=Ta[14]+273.15; Gn=G[12]
    # radiative cooling power at Ts=Ta noon, simple grey sky model
    return eps*sig*T**4*(1-esky)-(1-SR)*Gn
if __name__=='__main__':
    # baseline life-mean SR over 20y
    t=np.linspace(0,20,20001); tt=t%10
    SRb=0.65+0.20*np.exp(-tt/1.2)
    print('baseline life-mean SR (exp decay to 0.65, repaint 10y):',SRb.mean().round(3))
    SRb2=np.where(tt<3,0.85-0.17*tt/3,0.68-0.005*(tt-3))
    print('baseline linear 3y then slow:',SRb2.mean().round(3))
    cases={'Baseline aged paint':(0.70,0.90),
           'A claimed':(0.855,0.88),'A my est.':(0.80,0.87),
           'B clean/claimed aged':(0.87,0.94),'B fouled':(0.78,0.94),
           'C est.':(0.78,0.92),'D est.':(0.74,0.90),'E rainy est.':(0.78,0.92),
           'Ideal clean PDRC':(0.96,0.95)}
    for kind in ['hotdry','humid']:
      for el,Us in [('roof',[0.3,0.6,1.5,3.0]),('wall',[0.5,1.5,2.5])]:
        print(f'\n== {kind} {el} ==  daily gain kWh/m2 (and % vs baseline)')
        print('case'.ljust(24)+''.join(f'U={U:<12}' for U in Us))
        base={U:daily_gain(0.70,0.90,U,kind,el)[0] for U in Us}
        for n,(sr,e) in cases.items():
            row=n.ljust(24)
            for U in Us:
                g=daily_gain(sr,e,U,kind,el)[0]
                row+=f'{g:5.3f}({100*(g/base[U]-1):+4.0f}%) '
            print(row)
      print('\nnet noon cooling power at Ts=Ta', kind, {n:round(netcool(sr,e,kind)) for n,(sr,e) in cases.items()})
