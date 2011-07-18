#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil.h>
#include "params.h"
#include "common.h"
#include "bnd.h"
#include "GPU.h"
#include "Boundary.h"
#include "cosmo.h"
#include "Allocation.h"
#include "Io.h"
#include "Explicit.h"
#include "Atomic.h"
#ifdef WMPI
#include "communication.h"
#include "Interface.h"
#endif



//**********************************************************
//**********************************************************

extern "C" int Mainloop(int rank, int *pos, int *neigh, int ic_rank);

//**********************************************************
//**********************************************************


#define CUERR() //printf("\n %s on %d \n",cudaGetErrorString(cudaGetLastError()),ic_rank)

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define NCELLS3 (NCELLX+NBOUND2)*(NCELLY+NBOUND2)*(NCELLZ+NBOUND2)

#define N_INT 2048
#define A_INT_MAX 0.166667


//**********************************************************
//**********************************************************


int Mainloop(int rank, int *pos, int *neigh, int ic_rank)
{

  if(rank==0) printf("Mainloop entered by proc %d\n",rank);

  float tnext;


  dim3 blockion(NCELLX);           // USED BY IONISATION
  dim3 gridion(NCELLY,NCELLZ);

  dim3 bcool(BLOCKCOOL);           // USED BY COOLING
  dim3 gcool(GRIDCOOLX,GRIDCOOLY);
  
  dim3 blocksimple(NCELLX);        // USED BY ADVECTION THREADS
  dim3 gridsimple(NCELLY,NCELLZ);


#ifdef SDISCRETE
  int nthreadsource=min(nsource,128);
  dim3 gridsource((int)(round((float)(nsource)/float(nthreadsource))));
  dim3 blocksource(nthreadsource);
#endif

#ifndef WMPI

  dim3 blockboundx(NCELLY);
  dim3 gridboundx(NCELLZ);

  dim3 blockboundy(NCELLX);
  dim3 gridboundy(NCELLZ);

  dim3 blockboundz(NCELLX);
  dim3 gridboundz(NCELLY);

for (int igrp=0;igrp<NGRP;igrp++)
	{
  if(boundary==0) // transmissive boundary conditions
    {
      cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_yp<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_zp<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_ym<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_zm<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  else if(boundary==1) // reflexive boundary conditions
    {
      cusetboundaryref_zp<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_zm<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_yp<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_ym<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  else if(boundary==2) // Periodic boundary conditions
    {
      cusetboundaryper_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_yp<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_zp<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_ym<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_zm<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  else if(boundary==3) // Mixed boundary conditions
    {
      cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_yp  <<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_zp  <<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_ym  <<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_zm  <<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  }
#else
  dim3 blockboundx(NCELLY);
  dim3 gridboundx(NCELLZ);

  dim3 blockboundy(NCELLX);
  dim3 gridboundy(NCELLZ);

  dim3 blockboundz(NCELLX);
  dim3 gridboundz(NCELLY);


  if(neigh[5]!=rank)  
    {  
      exchange_zp(cuegy, cuflx, cuegy_new, buff, neigh, pos[2]%2);
      exchange_zm(cuegy, cuflx, cuegy_new, buff, neigh, pos[2]%2);
    }
  else
    {
      cusetboundaryper_zp<<<gridboundz,blockboundz>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      cusetboundaryper_zm<<<gridboundz,blockboundz>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
    }
  
  if(neigh[3]!=rank)
    {
      exchange_yp(cuegy, cuflx, cuegy_new, buff, neigh, pos[1]%2);
      exchange_ym(cuegy, cuflx, cuegy_new, buff, neigh, pos[1]%2);
    }
  else
    {
      cusetboundaryper_yp<<<gridboundy,blockboundy>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      cusetboundaryper_ym<<<gridboundy,blockboundy>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
    }

  if(neigh[1]!=rank)
    {
      exchange_xp(cuegy, cuflx, cuegy_new, buff, neigh, pos[0]%2);
      exchange_xm(cuegy, cuflx, cuegy_new, buff, neigh, pos[0]%2);
    }
  else
    {
      cusetboundaryper_xp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      cusetboundaryper_xm<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
    }


  if(boundary==0)
    {
      if(pos[0]==0) cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[1]==0) cusetboundarytrans_ym<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[2]==0) cusetboundarytrans_zm<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);

      if(pos[0]==(NGPUX-1)) cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[1]==(NGPUY-1)) cusetboundarytrans_yp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[2]==(NGPUZ-1)) cusetboundarytrans_zp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);

    }

#endif


#ifndef COSMO
  dt=courantnumber*dx/3./c;
  if(rank==0) printf("dx=%e cfl=%e dt=%e\n",dx,courantnumber,dt);
  tnext=t;//+ndumps*dt;
#else
  aexp=astart;
#ifndef FLAT_COSMO
  t=a2tgen(aexp,omegam,omegav,Hubble0);// Hubble0 in sec-1
#else
  t=a2t(aexp,omegav,Hubble0);// Hubble0 in sec-1
#endif
  
  tnext=t;
  float tstart=t;
  if(rank==0) printf("aexp= %f tstart=%f tmax=%f\n",aexp,t/unit_time,tmax/unit_time);

#ifndef FLAT_COSMO
  if(rank==0) printf("Building Expansion factor table");

  float da=(A_INT_MAX-aexp)/N_INT;
  float a_int[N_INT],t_int[N_INT];
  for(int i_int=0;i_int<N_INT;i_int++)
    {
      a_int[i]=aexp+i_int*da;
      t_int[i]=a2tgen(a_int[i],omegam,omegav,Hubble0); // Hubble0 in sec-1
    }

  int n_int=0;

#endif


#endif
  
  // some variables for field update
  int changefield=0;
  int forcedump;
  int ifield=0; // 1 because tfield stores the NEXT field
  float tfield;
  if(fieldlist){
    while(t>=tlist[ifield])
      {
	ifield++;
      }
    tfield=tlist[ifield];
    if(rank==0) printf("ICs (tstart=%f) between field #%d (t=%f) and field #%d (t=%f)\n",t/unit_time,ifield-1,tlist[ifield-1]/unit_time,ifield,tlist[ifield]/unit_time);
    if(rank==0) printf("starting with NEXT field #%d @ tfield =%f with tstart=%f\n",ifield,tlist[ifield]/unit_time,t/unit_time);// -1 because tfield stores the NEXT field
  }

  // log file
  FILE *logfile;
  if(rank==0) logfile=fopen("log.out","w");

#ifdef TIMINGS
  FILE *timefile;
  if(rank==0)
    {
      timefile=fopen("time.out","w");
    }
#endif

  //float ft=1./powf(2.,20);
  float ft=1.;
#ifdef COSMO
  float factfesc=1.;
#endif

float *factgrp;
factgrp=(float*)malloc(NGRP*sizeof(float));
FACTGRP;

  unsigned int timer;
  float q0=0.,q1=0.,q3;
#ifdef TIMINGS
  float q4,q7,q8,q9,q10,q11;
  double time_old,time_new;
#endif  
  if(rank==0)
    {
      cutCreateTimer(&timer);
      cutStartTimer(timer);

    }

  
  // MAIN LOOP STARTS HERE ======================================================>>>>
  // ============================================================================>>>>
  // ============================================================================>>>>
  // ============================================================================>>>>
  // ============================================================================>>>>
  // ============================================================================>>>>

  cudaThreadSynchronize();
#ifdef WMPI	  
  mpisynch();
#endif
  
  cuDumpResults(0,t,aexp,0);

  while(t<=tmax)
    {  
      

      cudaThreadSynchronize();
#ifdef WMPI	  
      get_elapsed(&time_old);
      mpisynch();
#endif
      if(rank==0)
	{
	  q3=q1-q0;
	  q0=cutGetTimerValue(timer);
	}
      

#ifndef COSMO
      dt=courantnumber*dx/3./c*ft;
      if(((nstep%ndisp)==0)&&(rank==0))
	{
	  printf(" ------------------ \n");
	  printf(" Step= %d Time= %f dt=%f tnext=%f cgpu (msec)=%f\n",nstep,t/unit_time,dt/unit_time,tnext/unit_time,q3);
	  printf(" ------------------ \n");
	}
#else
      dt=courantnumber*dx/3./c*ft;

      if(((nstep%ndisp)==0)&&(rank==0))
	{
	  printf(" ------------------------------\n");
	  printf(" Step= %d Time= %f Elapsed= %f dt= %f aexp=%f z=%f fesc=%f clump= %f Next tfield=%f cgpu=%f\n",nstep,t/unit_time,(t-tstart)/unit_time,dt/unit_time,aexp,1./aexp-1.,factfesc*fesc,clump,tfield/unit_time,q3);
	  printf(" ----------------------------- \n");
	  fprintf(logfile,"%d %f %f %f %f %f %f %f\n",nstep,t/unit_time,(t-tstart)/unit_time,dt/unit_time,aexp,1./aexp-1.,tfield/unit_time,q3);
	}
#endif
      

      if(fieldlist)
	{
	  // we must not go further than the next field
	  if(dt>=tfield-t)
	    {
#ifdef WMPI
	      if(rank==0) printf("last timestep with field #%d : next field= %f t=%f t+dt=%f\n",ifield,tfield/unit_time,t/unit_time,(t+dt)/unit_time);

	      if(((tfield-t)/unit_time)==0.)
		{
		  if(rank==0) printf("WARNING FIELD DT=O -> switch immediatly to next field\n"); 
		  cuGetField(ifield,ic_rank);
		  changefield=0;
		  ifield++;
		  tfield=tlist[ifield];
		  ft=1./powf(2.,20);
		}
	      else
		{
		  changefield=1;
		  dt=tfield-t;
		  if(rank==0) printf("dt set to %f\n",dt/unit_time);
		}
#else
	      if(rank==0) printf("last timestep with field #%d : next field= %f t=%f t+dt=%f\n",ifield,tfield/unit_time,t/unit_time,(t+dt)/unit_time);

	      if(((tfield-t)/unit_time)==0.)
		{
		  if(rank==0) printf("WARNING FIELD DT=O -> switch immediatly to next field\n"); 
		  cuGetField(ifield,ic_rank);
		  changefield=0;
		  ifield++;
		  tfield=tlist[ifield];
		  ft=1./powf(2.,20);
		}
	      else
		{
		  changefield=1;
		  dt=tfield-t;
		  if(rank==0) printf("dt set to %f\n",dt/unit_time);
		}
#endif
	    }
	}

      //================================== UNSPLIT 3D SCHEME=============================


	for (int igrp=0;igrp<NGRP;igrp++)
		{
		#ifdef COSMO
		      cuComputeELF<<<gridsimple,blocksimple>>>(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cusrc0, cuegy_new+igrp*NCELLS3, c, dx, dt, nstep,aexp);
		#else
		      cuComputeELF<<<gridsimple,blocksimple>>>(cuegy+igrp*NCELLS3, cuflx+igrp*NCELLS3*3, cusrc0, cuegy_new+igrp*NCELLS3, c, dx, dt, nstep,1.);
		#endif
		
		      cudaThreadSynchronize();
		      CUERR();
			      if(verbose) puts("Hyperbolic Egy ok");
		
		#ifdef COSMO
		      cuComputeF_TOTAL_LF<<<gridsimple,blocksimple>>>(cuflx+igrp*NCELLS3*3,cudedd,cusrc0,cuflx_new+igrp*NCELLS3*3,c,dx,dt,nstep,cuegy+igrp*NCELLS3, aexp);
		#else
		      cuComputeF_TOTAL_LF<<<gridsimple,blocksimple>>>(cuflx+igrp*NCELLS3*3,cudedd,cusrc0,cuflx_new+igrp*NCELLS3*3,c,dx,dt,nstep,cuegy+igrp*NCELLS3,1.);
		#endif
		      cudaThreadSynchronize();
		      CUERR();
	
		#ifdef SDISCRETE
		#ifdef COSMO
		      if(kf!=0.) factfesc=exp(kf-powf(aexp/a0,af));
		      cuAddSource<<<gridsource,blocksource>>>(cuegy_new+igrp*NCELLS3,cuflx_new+igrp*NCELLS3*3,cusrc0,cusrc0pos,dt*fesc*factfesc*factgrp[igrp],dx,nsource,aexp,c);
		#else
		      cuAddSource<<<gridsource,blocksource>>>(cuegy_new+igrp*NCELLS3,cuflx_new+igrp*NCELLS3*3,cusrc0,cusrc0pos,dt*fesc*factgrp[igrp],dx,nsource,1.,c);
		#endif
		
		      CUERR();
		      if(verbose) puts("Add Source ok");
		#endif
		
		      if(verbose) puts("Hyperbolic Flux ok");
		
		      cudaThreadSynchronize();

		}

#ifdef TIMINGS     
#ifdef WMPI	  
      mpisynch();
#endif
      if(rank==0)
	{
	  q11=cutGetTimerValue(timer);
	}
#endif
	
#ifdef TESTCOOL  
#ifdef COSMO
      cuComputeIon<<<gridion,blockion>>>(cuegy_new, cuflx_new, cuxion, cudensity, cutemperature, dt/cooling, c, egy_min,unit_number,aexp);
#else
      cuComputeIon<<<gridion,blockion>>>(cuegy_new, cuflx_new, cuxion, cudensity, cutemperature, dt/cooling, c, egy_min,unit_number,1.);
#endif
#endif
      CUERR();
      if(verbose) puts("Chemistry     ok");
      cudaThreadSynchronize();
#ifdef WMPI
      mpisynch();
#endif

#ifdef TIMINGS
      if(rank==0)
	{
	  q4=cutGetTimerValue(timer);
	}
#endif

	  // Here cuegy is used to store the temperature
#ifdef COSMO
      float hubblet=Hubble0*sqrtf(omegam/aexp+omegav*(aexp*aexp))/aexp;
      cuComputeTemp<<<gcool,bcool>>>( cuxion, cudensity, cutemperature, cuegy_new, fudgecool, c, dt/cooling, unit_number, ncvgcool, aexp, hubblet, cuflx_new, clump);
#else
      cuComputeTemp<<<gcool,bcool>>>( cuxion, cudensity, cutemperature, cuegy_new, fudgecool, c, dt/cooling, unit_number, ncvgcool, 1.,   0., cuflx_new, clump);
#endif
      CUERR();
      if(verbose) puts("Cooling  ok");
      cudaThreadSynchronize();
#ifdef WMPI	  
      mpisynch();
#endif

#ifdef TIMINGS
      cudaThreadSynchronize();
#ifdef WMPI
      mpisynch();
#endif
      if(rank==0)
	{
	  q8=cutGetTimerValue(timer);
	}
#endif

      cudaMemcpy(cuegy,cuegy_new,NCELLS3*sizeof(float)*NGRP,cudaMemcpyDeviceToDevice);
      cudaMemcpy(cuflx,cuflx_new,NCELLS3*sizeof(float)*3*NGRP,cudaMemcpyDeviceToDevice);


#ifdef TIMINGS
      cudaThreadSynchronize();
#ifdef WMPI
      mpisynch();
#endif
      if(rank==0)
	{
	  q10=cutGetTimerValue(timer);
	}
#endif


      if(verbose) puts("Dealing with boundaries");


#ifndef WMPI
for (int igrp=0;igrp<NGRP;igrp++)
	{
  if(boundary==0) // transmissive boundary conditions
    {
      cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NGRP*3);
      cusetboundarytrans_yp<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NGRP*3);
      cusetboundarytrans_zp<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NGRP*3);
      cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NGRP*3);
      cusetboundarytrans_ym<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NGRP*3);
      cusetboundarytrans_zm<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NGRP*3);
    }
  else if(boundary==1) // reflexive boundary conditions
    {
      cusetboundaryref_zp<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_zm<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_yp<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_ym<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  else if(boundary==2) // Periodic boundary conditions
    {
      cusetboundaryper_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_yp<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_zp<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_ym<<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryper_zm<<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  else if(boundary==3) // Mixed boundary conditions
    {
      cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_yp  <<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_zp  <<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_ym  <<<gridboundy,blockboundy>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
      cusetboundaryref_zm  <<<gridboundz,blockboundz>>>(cuegy+igrp*NCELLS3, cuxion, cudensity, cutemperature, cuflx+igrp*NCELLS3*3);
    }
  
}
#else

  
  
  if(neigh[5]!=rank)
    {  
      exchange_zp(cuegy, cuflx, cuegy_new, buff, neigh, pos[2]%2);
      exchange_zm(cuegy, cuflx, cuegy_new, buff, neigh, pos[2]%2);
    }
  else
    {
      cusetboundaryper_zp<<<gridboundz,blockboundz>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      cusetboundaryper_zm<<<gridboundz,blockboundz>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
    }

  if(neigh[3]!=rank)
    {
      exchange_yp(cuegy, cuflx, cuegy_new, buff, neigh, pos[1]%2);
      exchange_ym(cuegy, cuflx, cuegy_new, buff, neigh, pos[1]%2);
    }
  else
    {
      cusetboundaryper_yp<<<gridboundy,blockboundy>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      cusetboundaryper_ym<<<gridboundy,blockboundy>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
    }

  if(neigh[1]!=rank)
    {
      exchange_xp(cuegy, cuflx, cuegy_new, buff, neigh, pos[0]%2);
      exchange_xm(cuegy, cuflx, cuegy_new, buff, neigh, pos[0]%2);
    }
  else
    {
      cusetboundaryper_xp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      cusetboundaryper_xm<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
    }

  if(boundary==0)
    {
      //printf("coucou\n");
      if(pos[0]==0) cusetboundarytrans_xm<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[1]==0) cusetboundarytrans_ym<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[2]==0) cusetboundarytrans_zm<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);

      if(pos[0]==(NGPUX-1)) cusetboundarytrans_xp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[1]==(NGPUY-1)) cusetboundarytrans_yp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);
      if(pos[2]==(NGPUZ-1)) cusetboundarytrans_zp<<<gridboundx,blockboundx>>>(cuegy, cuxion, cudensity, cutemperature, cuflx);

    }

#endif

  cudaThreadSynchronize(); 
#ifdef WMPI
  mpisynch();
#endif

#ifdef TIMINGS
  if(rank==0)
    {
      q7=cutGetTimerValue(timer);
    }
#endif
  
  //printf("proc %d ready to dump\n",ic_rank);

  if(((nstep%ndumps)==0)||(forcedump))
	{
	  ntsteps=ntsteps+1;
	  forcedump=0;
#ifdef COSMO
#ifdef FLAT_COSMO
	  float aexpdump=t2a(t+dt,omegav,Hubble0);
#else
	  if(t+dt>t_int_max)
	    {
	      aexpdump=(a_int[int_step+2]-a_int[int_step+1])/(t_int[int_step+2]-t_int[int_step+1])*(t+dt-t_int[int_step+1]);
	    }
	  else
	    {
	      aexpdump=(a_int[int_step+1]-a_int[int_step])/(t_int[int_step+1]-t_int[int_step])*(t+dt-t_int[int_step]);
	    }
#endif
	  cuDumpResults(ntsteps,t+dt,aexpdump,ic_rank);
#else
	  cuDumpResults(ntsteps,t+dt,0.,ic_rank);
#endif
	  tnext=tnext+ndumps*dt/ft;
	  if(rank==0) printf("tnext=%f\n",tnext/unit_time);
	}

      //--------------------------------------------------------------------
      // Dealing with fieldlists
      //--------------------------------------------------------------------

      ft=fminf(ft*2.,1.);
      
      if(fieldlist)
	{
	  if(changefield)
	    {
	    int ercode;
#ifdef WMPI
	      ercode=cuGetField(ifield,ic_rank);
#else
	      ercode=cuGetField(ifield,0);
#endif
	      if(ercode==38)
		{
		  if(rank==0)
		    {
		      fclose(logfile);
		      fclose(timefile);
		    }
		  abort();
		}
	      forcedump=0;
	      changefield=0;
	      ifield++;
	      tfield=tlist[ifield];
	      ft=1./powf(2.,20);
	      //ft=1.;
	    }
	}


      // UPDATING VARIABLES

      t=t+dt;
      if(t>tmax)
	{
	  puts("t > tmax -----> run will be terminated");
	}
#ifdef COSMO

#ifdef FLAT_COSMO
      aexp=t2a(t,omegav,Hubble0); // A CHANGER PAR INTERPOLATION
#else
      if(t>t_int_max)
	{
	  int_step++;
	}
      aexp=(a_int[int_step+1]-a_int[int_step])/(t_int[int_step+1]-t_int[int_step])*(t-t_int[int_step]);
#endif


      c=c_r/aexp;
#endif       
      
      cudaThreadSynchronize();
#ifdef WMPI
      mpisynch();
#endif
      if(rank==0)
	{
	  q1=cutGetTimerValue(timer);
	}


      nstep++;
      if(nstep==nmax) {
	if(rank==0) puts("Max number of steps achieved: STOP");
	break;
      }

      cudaThreadSynchronize();
#ifdef WMPI
      get_elapsed(&time_new);
      time_new=time_new-time_old;
      mpireducemax(&time_new);
      mpisynch();
#endif

#ifdef TIMINGS
      if(rank==0){
	q9=cutGetTimerValue(timer);
	printf("transport=%f chem=%f cool=%f update=%f bound=%f IO=%f,grand total=%f time_new=%lf\n",q11-q0,q4-q11,q8-q4,q10-q8,q7-q10,q9-q7,q9-q0,time_new);
	fprintf(timefile,"%d %f %f %f %f %f %f %f\n",nstep-1,q11-q0,q4-q11,q8-q4,q10-q8,q7-q10,q9-q7,q9-q0,time_new);
      }


#endif

      cudaThreadSynchronize();
#ifdef WMPI	  
      mpisynch();
#endif

    }

  if(rank==0) fclose(logfile);
#ifdef TIMINGS
  if(rank==0) fclose(timefile);
#endif
  return 0;
}

