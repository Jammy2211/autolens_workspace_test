pro plt_hpsi,n_plot,m_plot,twod=twod,real=real,imag=imag,ps=ps,yrange=yrange

; Written July 2010 by Joel Berge

; Purpose
;  plot 1D or 2D Exponential Shapelets
;
; Inputs
;  n_plot - vector of n's to plot
;  m_plot - vector of m's to plot
;
; Optional keyword
;  yrange - controls the amplitude of the plot
;
; Keyword
;  /twod - plot 2D basis functions (default: plot 1D radial functions)
;  /real - plot real part of 2D basis functions
;  /imag - plot imaginary part of 2D basis functions
;  /ps - plot in a postscript file

; usage : plt_hpsi,[0,1,2],[0,1,2]
;       or plt_hpsi,[0,1,2],[0,-1,0,1,-2,-1,0,1,2],/twod


if keyword_set(ps) then ops,file='hlets.ps',/color,bits_per_pixel=17
if keyword_set(twod) then loadct,1 else tek_color
nr=100 & nphi=100
r=[xgen(0,100,np=nr-1),100]
phi=[xgen(0,2*!pi,np=nphi-1),2*!pi]
nvec=[0,1,2,3,4,5,6] & nn=n_elements(nvec)
beta=1.
if not keyword_set(yrange) then yrange=[-0.3,0.3]

nx=201 & ny=201
ic=fix(nx/2) & jc=fix(ny/2)
r2d=fltarr(nx,ny)
phi2d=fltarr(nx,ny)
center=[ic,jc]
for i=0,nx-1 do begin
   for j=0,ny-1 do begin
      r2d[i,j]=sqrt((i-center[0])^2+(j-center[1])^2)
      phi2d[i,j]=atan((j-center[1]),(i-center[0]))
   endfor
endfor
neg=where(phi2d lt 0)
phi2d[neg]+=2*!pi

if n_elements(n_plot) eq 0 then n_plot=nvec
;if n_elements(m_plot) eq 0 then m_plot=nvec

if keyword_set(twod) then begin
   xpanels=max(n_plot)+1 ;number of ns }irrespective of which ones to plot
   ypanels=2*max(n_plot)+1 ;number of ms   }
   if not keyword_set(ps) then begin
      frx1=0.1
      frx2=0.97
      fry1=0.07
      fry2=0.97
   endif else begin
      frx1=0.21
      frx2=0.94
      fry1=0.14
      fry2=0.93
   endelse
   width=(frx2-frx1)/xpanels
   height=(fry2-fry1)/ypanels
endif

firstplot=1
nnp=n_elements(n_plot)
nplots=0
for i=0,nnp-1 do begin
   nplotsi=0
   color=i
   if i gt 0 then color+=1
   if n_elements(m_plot) eq 0 then mvec=indgen(2*i+1)-i else mvec=m_plot
   nm=n_elements(mvec)
   for j=0,nm-1 do begin
      n=n_plot[i] & m=m_plot[j]
      if(abs(m) gt n) then continue
      if not keyword_set(twod) then begin
         psi=hpsi(r,0,n,m,beta,/rad)
         if nplots eq 0 then begin
            plot,r,psi,xtitle='r',ytitle=textoidl('R_{nm}'),yrange=yrange
            leg_txt='(n,m)=('+strtrim(n,1)+','+strtrim(m,1)+')'
            leg_lst=0
            leg_col=color
         endif else begin
            if nplotsi eq 0 then begin
               oplot,r,psi,color=color
               leg_lst=[leg_lst,0]
            endif else begin
               oplot,r,psi,linestyle=nplotsi+1,color=color
               leg_lst=[leg_lst,nplotsi+1]
            endelse
            leg_txt=[leg_txt,'(n,m)=('+strtrim(n,1)+','+strtrim(m,1)+')']
            ;leg_lst=[leg_lst,nplotsi+1]
            leg_col=[leg_col,color]
         endelse
      endif else begin
         if keyword_set(real) then title=textoidl('Re(\psi_{nm})') else $
            if keyword_set(imag) then title=textoidl('Im(\psi_{nm})') else $
               title=textoidl('|\psi_{nm}|')
         if firstplot eq 1 then $
            plot,[0],[0],xrange=[-0.5,max(n_plot)+0.5],yrange=[-max(n_plot)-0.5,max(n_plot)+0.5],/xstyle,/ystyle,title=title,xtitle='n',ytitle='m',xtickname=xtickname,ytickname=ytickname
         firstplot=0
         psi=complexarr(nx,ny)
        for k=0,nx-1 do $
           for kk=0,ny-1 do $
              psi[k,kk]=hpsi(r2d[k,kk],phi2d[k,kk],n,m,beta)
         ;plt_image,real_part(psi),/fr,title='n='+strtrim(n,1)+', m='+strtrim(m,1)
         pos=[frx1+n*width,fry1+(m+max(n_plot))*height,frx1+(n+1)*width,fry1+(m+max(n_plot)+1)*height]
         if keyword_set(real) then $
            plt_image,real_part(psi),position=pos,/noerase,cran=yrange else $
               if keyword_set(imag) then $
                  plt_image,imaginary(psi),position=pos,/noerase,cran=yrange else $
                     plt_image,abs(psi),position=pos,/noerase,cran=yrange
      endelse
      nplotsi+=1
      nplots+=1
   endfor
endfor
if not keyword_set(twod) then legend,leg_txt,psym=0,linestyle=leg_lst,box=0,/right,color=leg_col,charsize=1.3

if keyword_set(ps) then cps


end
