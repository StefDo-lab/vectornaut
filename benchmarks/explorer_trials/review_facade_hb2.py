from hb import *
def addR(U,R): return 1/(1/U+R)
print("Insulation equivalence: added R (m2K/W) giving same % cut as SR 0.70->X (roof, hotdry)")
for U in [0.3,0.6,1.5,3.0]:
    b=daily_gain(0.70,0.90,U)[0]
    out=[]
    for SR in [0.80,0.855,0.90]:
        g=daily_gain(SR,0.90,U)[0]
        # find R
        lo,hi=0,5
        for _ in range(60):
            m=(lo+hi)/2
            if daily_gain(0.70,0.90,addR(U,m))[0]>g: lo=m
            else: hi=m
        out.append(f"SR{SR}: dR={m:.3f} (~{m*0.035*1000:.0f} mm mineral wool)")
    print(f"U={U}:",'; '.join(out))
print()
for kind in ['hotdry','humid']:
  print("==",kind,"walls ==")
  for U in [0.5,1.5,2.5]:
    b=daily_gain(0.70,0.90,U,kind,'wall')[0]
    def p(g): return f"{100*(g/b-1):+.0f}%"
    F=daily_gain(0.75,0.90,addR(U,0.17),kind,'wall')[0]
    F_ins=daily_gain(0.70,0.90,addR(U,0.17),kind,'wall')[0]
    G=daily_gain(0.60,0.90,addR(U,1.4),kind,'wall')[0]
    G_eps=daily_gain(0.60,0.90,addR(U,1.4),kind,'wall')[0]
    G_whitepaint=daily_gain(0.70,0.90,addR(U,1.4),kind,'wall')[0]
    print(f"U={U}: F(glazed SR.75 + sealed gap R.17) {p(F)} | same gap w/ baseline paint {p(F_ins)} | G cork 60mm R1.4 aged render SR.60 {p(G)} | 60mm EPS+cool paint {p(G_whitepaint)}")
# H: shading screen: reduce wall-incident beam by f, add hot screen convective/radiative coupling approx: extra 10% of absorbed-by-screen reaching wall
print()
h,Ta,G,esky=climate('hotdry')
for f in [0.5,0.7,0.85]:
    for U in [1.5]:
        b=daily_gain(0.70,0.90,U,'hotdry','wall')
        # model: effective solar on wall = (1-f)*Gw + 0.15*f*Gw*alpha_screen(0.5)
        import hb
        orig=hb.wall_G
        hb.wall_G=lambda hh,orig=orig,f=f: orig(hh)*((1-f)+0.15*f*0.5)
        s=daily_gain(0.70,0.90,U,'hotdry','wall',hc=5)  # still air trap lowers hc
        hb.wall_G=orig
        print(f"screen shading {f}: 24h {100*(s[0]/b[0]-1):+.0f}%  peak {100*(s[1]/b[1]-1):+.0f}%")
