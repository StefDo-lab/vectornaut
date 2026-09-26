import math
nu=1.05e-6; rho=1025
U=7.7  # 15 kn
for Cf in (0.0015,0.0020):
    ut=U*math.sqrt(Cf/2); lp=nu/ut; Up=1/math.sqrt(Cf/2)
    print(f"Cf={Cf}: u_tau={ut:.3f} m/s, l+={lp*1e6:.2f} um, U+={Up:.1f}, DR per dU+=1: {100*(1-(Up/(Up+1))**2):.1f}%, tau_w={0.5*rho*U**2*Cf:.1f} Pa")
ut=U*math.sqrt(0.0017/2); lp=nu/ut; Up=1/math.sqrt(0.0017/2)
# C: grooved LIS effective slip, pitch L, fraction phi (Philip perfect slip, longitudinal)
for L,phi in ((20e-6,0.5),(40e-6,0.8),(80e-6,0.8)):
    bx=(L/math.pi)*math.log(1/math.cos(math.pi*phi/2)); bz=bx/2
    dU=(bx-bz)/lp  # ~ virtual origin difference (Luchini), upper bound (perfect slip)
    print(f"C: L={L*1e6:.0f}um phi={phi}: bx+={bx/lp:.2f}, dU+~{dU:.2f}, DR~{100*(1-(Up/(Up+dU))**2):.1f}% (upper bound, perfect slip, no finite cells); groove w+={L*phi/lp:.1f}")
# lubricant dissolution in turbulent flow: Shaw-Hanratty k+ = 0.0889 Sc^-0.704
D=8e-10; Sc=nu/D; k=0.0889*Sc**-0.704*ut
print(f"mass-transfer k={k:.2e} m/s (Sc={Sc:.0f})")
for name,csat,rhol in (("n-heptane",3.4e-3,684),("HMDSO 0.65cSt",0.93e-3,760),("octamethyltrisiloxane L3 1cSt",3.4e-5,820)):
    inv=5e-6*rhol  # 10 um deep, 50% area -> 5 um equivalent film
    flux=k*csat
    print(f"{name}: flux={flux*3600*24*1e3:.2f} g/m2/day underway, inventory={inv*1e3:.1f} g/m2, lasts {inv/flux/3600:.1f} h; 10,000 m2 hull loss {flux*1e4*86400:.1f} kg/day")
# E: electrolytic gas vs dissolution
for i in (0.02,1.0):
    n=i/(2*96485)  # mol H2 /m2/s
    print(f"E: j={i} A/m2 -> H2 {n:.2e} mol/m2/s, film growth {n*0.0245/2*1e9:.2f} nm/s at 2 bar")
kg=0.0889*(nu/2e-9)**-0.704*ut
Csat_H2=7.8e-6*2e5
print(f"E: H2 dissolution loss ~{kg*Csat_H2:.2e} mol/m2/s (k={kg:.1e}); needs j~{kg*Csat_H2*2*96485:.0f} A/m2")
for I in (300,1000):
    print(f"ICCP {I} A -> Cl2 {I/(2*96485)*70.9*86400/1000:.0f} kg/day")
# D: lamella energy release rate for a debond of length a under wall shear
E=1e6;h=100e-6
for tau in (15,30):
    for a in (0.1,1.0):
        print(f"D: tau={tau}Pa a={a}m G={(tau*a)**2/(2*E*h):.3f} J/m2")
# G: crust bending length on Winkler gel
for Ec,hc,Eg,tg in ((50e6,30e-6,0.1e6,300e-6),(10e6,30e-6,0.1e6,300e-6)):
    Dp=Ec*hc**3/(12*(1-0.25)); kw=Eg/tg*3  # incompressible thin layer ~ factor
    print(f"G: crust E={Ec/1e6}MPa h={hc*1e6}um -> bending length {(Dp/kw)**0.25*1e3:.3f} mm")
# H: release stress ratio sqrt(E/t)
ref=math.sqrt(1.0/0.25); new=math.sqrt(0.1/0.35)
print(f"H: sqrt(E/t) ratio new/ref = {new/ref:.2f}")
