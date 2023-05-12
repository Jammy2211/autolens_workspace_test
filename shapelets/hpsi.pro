function hpsi,r,phi,n,m,beta,radial=radial

; Written July 2010 by Joel Berge

; compute psi function
;
; Inputs
;  r,phi - arguments of the function
;  n,m - indices
;  beta - beta
;
; Keyword
;  /radial - plot radial part (default: plot 2D fucntions)

imath=complex(0,1)
betan=beta*(n+1./2)
pref1=(1./betan)^(1+abs(m))
pref2=sqrt(factorial(n-abs(m))/((2.*n+1)*factorial(n+abs(m))))
lag=laguerre(r/betan,n-abs(m),2*abs(m))
rnm=(-1.)^n*pref1*pref2*lag*r^(abs(m))*exp(-r/(2*betan))

expphi=exp(-imath*m*phi)

psi=rnm*expphi

if keyword_set(radial) then return,rnm else return,psi

end
