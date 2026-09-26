def brentq(f,a,b):
    for _ in range(200):
        m=(a+b)/2
        if f(a)*f(m)<=0: b=m
        else: a=m
    return (a+b)/2
import numpy as np
s=5.67e-8
def Tsurf(a,eps,I,Ta,Lsky,hc=10,Fsky=1.0,Tg=None):
    # steady surface balance, adiabatic back
    Tak=Ta+273.15
    Tgk=(Tg+273.15) if Tg else Tak
    f=lambda T: a*I+eps*Fsky*Lsky+eps*(1-Fsky)*s*Tgk**4 - eps*s*T**4 - hc*(T-Tak)
    return brentq(f,250,400)-273.15
Ta=38; Lsky=0.78*s*(Ta+273.15)**4  # dry desert sky
print("C2 roof aged SR0.70 e0.9:",Tsurf(0.30,0.90,900,Ta,Lsky))
print("C2 roof SR0.95 e0.96:",Tsurf(0.05,0.96,900,Ta,Lsky))
print("C2 facade SR0.95 e0.96, I=600, half ground at 55C:",Tsurf(0.05,0.96,600,Ta,Lsky,Fsky=0.5,Tg=55))
print("C6 aged SR0.72 roof:",Tsurf(0.28,0.9,900,Ta,Lsky))
# daily mean sol-air gains
def gain(U,a,Imean,dR,eps,dT=8,h=20): return U*(dT+a*Imean/h-eps*dR/h)
for lab,U,Im,dR in [("roof U3",3,312,100),("W wall U3.4",3.4,175,40)]:
    g0=gain(U,0.30,Im,dR,0.9); 
    for a in (0.13,0.15,0.04):
        g=gain(U,a,Im,dR,0.95); print(lab,"a=0.30->",a, round(g0,1),round(g,1),f"{100*(1-g/g0):.0f}%")
g0=gain(3.4,0.35,175,40,0.9); g=gain(3.4,0.04,175,40,0.96); print("C3 wall",round(g0,1),round(g,1),f"{100*(1-g/g0):.0f}%")
# C5 louver temperature rise
for h in (8,15,25,30):
    print("louver dT for 130/260 W/m2 at h_total",h, 130/h,260/h)
print("low-e underside radiative to wall, dT=7K, eps_eff 0.1-0.2:",[e*4*s*(310)**3*7 for e in (0.1,0.2)])
