pro ortho_hpsi,rad=rad

; look at psi orthogonality

nr=100000
rmax=1d11
r=[xgen(0,rmax,np=nr-1,/double),rmax]
nphi=1000
phi=[xgen(0,2*!pi,np=nphi-1),2*!pi]
nvec=[0,1,2,3,4,5,6] & nn=n_elements(nvec)
beta=1.

if keyword_set(rad) then $
   psi=dblarr(nr,nn,2*nn+1) else $
      psi=complexarr(nr,nphi,nn,2*nn+1)
nm_arr=intarr(2,nn,2*nn+1)
for i=0,nn-1 do for j=0,2*nn do nm_arr[*,i,j]=[-156,-156]
for i=0,nn-1 do begin
   if i gt 0 then mvec=indgen(2*i+1)-i else mvec=0
   nm=n_elements(mvec)
   print,mvec
   for j=0,nm-1 do begin
      n=nvec[i] & m=mvec[j]
      nm_arr[0,i,j]=n
      nm_arr[1,i,j]=m
      if keyword_set(rad) then psi[*,i,j]=hpsi(r,0,n,m,beta,/rad) else begin
         for k=0,nphi-1 do psi[*,k,i,j]=hpsi(r,phi[k],n,m,beta)
      endelse
   endfor
endfor

openw,lunw,'aortho.txt',/get_lun
for i1=1,nn-1 do begin
   if i1 gt 0 then m1vec=indgen(2*i1+1)-i1 else m1vec=0
   nm1=n_elements(m1vec)
   for j1=0,nm1-1 do begin
      for i2=0,nn-1 do begin
         if i2 gt 0 then m2vec=indgen(2*i2+1)-i2 else m2vec=0
         nm2=n_elements(m2vec)
         for j2=0,nm2-1 do begin
            n1=nm_arr[0,i1,j1] & m1=nm_arr[1,i1,j1]
            n2=nm_arr[0,i2,j2] & m2=nm_arr[1,i2,j2]
            if n1 eq -156 or m1 eq -156 or n2 eq -156 or m2 eq -156 then print,'coucou'
            
            if keyword_set(rad) then begin
               psi1=psi[*,i1,j1] & psi2=psi[*,i2,j2]
               prod=psi1*psi2
               pscal=int_tabulated(r,prod,/double)
               printf,lunw,format='(%"n1,m1=%f,%f, n2,m2=%f,%f, int=%f")',n1,m1,n2,m2,pscal
            endif else begin
               psi1=psi[*,*,i1,j1] & psi2=psi[*,*,i2,j2]
               prod=conj(psi1)*psi2
               pscalrad_re=dblarr(nphi)
               pscalrad_im=dblarr(nphi)
               for k=0,nphi-1 do begin
                  pscalrad_re[k]=int_tabulated(r,real_part(prod[*,k]),/double)
                  pscalrad_im[k]=int_tabulated(r,imaginary(prod[*,k]),/double)
               endfor
               pscal_re=int_tabulated(phi,pscalrad_re,/double)
               pscal_im=int_tabulated(phi,pscalrad_im,/double)
               pscal=complex(pscal_re,pscal_im)
               printf,lunw,format='(%"n1,m1=%f,%f, n2,m2=%f,%f, int=%f+i%f")',n1,m1,n2,m2,real_part(pscal),imaginary(pscal)
            endelse
            ;print,'(n1,m1)='+strtrim(n1,1)+','+strtrim(m1,1)+',  (n2,m2)='+strtrim(n2,1)+','+strtrim(m2,1)+',  pscal='+strtrim(pscal,1)
         endfor
         printf,lunw,format="(A)",' '
      endfor
   endfor
endfor
close,lunw
free_lun,lunw

end
