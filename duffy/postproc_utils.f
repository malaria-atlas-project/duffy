! Author: Anand Patil
! Date: 6 Feb 2009
! License: Creative Commons BY-NC-SA
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      SUBROUTINE duffy_postproc(fb,f0,p1,nx,cmin,cmax)

cf2py intent(inplace) fb
cf2py integer intent(in), optional :: cmin = 0
cf2py integer intent(in), optional :: cmax = -1
cf2py intent(hide) nx
cf2py threadsafe

      DOUBLE PRECISION fb(nx), f0(nx), p1, pb, p0
      INTEGER nx, i, cmin, cmax

      EXTERNAL DSCAL

      if (cmax.EQ.-1) then
          cmax = nx
      end if


        do i=cmin+1,cmax
            pb = 1.0D0 / (1.0D0 + dexp(-fb(i)))
            p0 = 1.0D0 / (1.0D0 + dexp(-f0(i)))
            pb = (pb*p0 + (1-pb)*p1)
            fb(i) = pb*pb
        end do


      RETURN
      END