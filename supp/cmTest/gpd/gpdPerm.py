# Original code by: Theo Knijnenburg, Institute for Systems Biology, Jan 6 2009
# Tranfered from matlab to python by: Ryan Tasseff, Institute for Systems Biology, Dec 2011

import numpy as np
import scipy as sp
import scipy.interpolate as interp
import scipy.optimize as opt
from scipy import arange, array, exp
import os

# will need this data for fit assesment

# This will be called from other directories
# assume the needed files are in the directory 
# this module is in: I need that path
dirname = os.path.dirname(__file__)
dirpath = os.path.abspath(dirname)


f = open(dirpath+"/k.dat","r")
data = [map(float,line.split()) for line in f]
f.close()
_ktable = np.array(data)[::-1,0]	

f = open(dirpath+"/p.dat","r")
data = [map(float,line.split()) for line in f]
f.close()
_ptable = np.array(data)[0,:]

f = open(dirpath+"/W2.dat","r")
data = [map(float,line.split()) for line in f]
f.close()
_W2table = np.array(data)[::-1,:]

f = open(dirpath+"/A2.dat","r")
data = [map(float,line.split()) for line in f]
f.close()
_A2table = np.array(data)[::-1,:]


del f
del data
del dirname

def est(x0,y,Nexcmax=250,alpha=0.05,method='ML'):
	"""Estimates the p value from a permutation test
	using a generalized pereto dist to approximate the 
	tail of the test statistic distribution.
	
	x0 	test statistic
	y	permutation values 
	Nexcmax	number of initial exceedances (defult: 250)
	alpha	not yet implimented 
	method	maximum liklyhood ('ML') only at this time
	Returns:
	p 	estimate of p values
	"""
	if method!= 'ML':
		raise ValueError("only method=ML is currently implimented")
	
	y=np.sort(y)[::-1]
	N=len(y)
	M=np.sum(y>=x0)
	Nexcmax = min(Nexcmax,N/4.0) #assmue tail is smaller then a quarter
	if M>=10:
		p = float(M)/float(N)
	else:
		Nexcvec = range(int(np.floor(Nexcmax)),0,-10)
		LNV = len(Nexcvec)
		count = 0
		p = np.nan
		while np.isnan(p) and count<LNV:
			Nexc = Nexcvec[count]
			p = pgpdML(y,x0,float(N),Nexc)
			count = count+1

	return p

def pgpdML(y,x0,N,Nexc):
	"""Computing permutation test P-value of the 
	GPD approximation.
	x0         original statistic
	y          permutation values
	N          number of permutation values
	Nexc       number of permutations used to approximate the tail
	Returns
	Phat       estimated P-value
	"""
	Padth = 0.05
	# define tail 
	z = y[:Nexc]
	t = np.mean(y[Nexc-1:Nexc+1])
	z = z-t
	frac = float(Nexc)/float(N)
	# fitting tail
	parmhat = gpdfit(z)
	Phat = np.nan
	# *** check if fit was a sucessfull
	if (not np.isnan(parmhat[0])) and (not np.isnan(parmhat[1])):
		a = parmhat[0]
		k = parmhat[1]
		# get the cdf 
		cdfV = gpdcdf(z,a,k)
		Pad = gpdgoft(cdfV,k)	
 
		if Pad > Padth:
			Phat = gpdPval(x0-t,parmhat)
			Phat = frac*Phat
		
	return Phat


def gpdfit(z):
	"""Fitting the generalized pareto distribution using ML
	z       exceedances
	return
	parmhat estimated shape and scale parameter
	Code based on:
	Hosking and Wallis, Technometrics, 1987
	Grimshaw, Technometrics, 1993
	Matlab gpfit.m

	"""
	TolBnd = 1E-10
	parmhat = np.array([1,0])

	parmhat= opt.fmin(_nll,parmhat,args=(z,),xtol=1E-10,ftol=1E-10,disp=False)

	# check boundry
	if np.abs(parmhat[1]-1)<TolBnd:
		parmhat = np.array([np.nan,np.nan])

	return parmhat


	


def _nll(parmhat,z):
	# trouble passing arguments
	a = parmhat[0]
	k = parmhat[1]
	n = float(len(z))
	m = float(max(z))
	
	if k>1:
		L = -1E52
	elif a<=max(1E-21,np.abs(k*m)): # constrained space A in Grimshaw 		
		L = -1E52
	else:
		if np.abs(k)<1E-52:
			L = -n*np.log(a)-(1.0/a)*np.sum(z)
		else:
			L = -n*np.log(a)+(1.0/k-1.0)*np.sum(np.log(1.0-k*z/a))
	
	return -L
	

def gpdcdf(x,a,k):
	"""1 - CDF of the generalized pareto distribution
	x       exceedances
	k       shape parameter
	a       scale parameter
	return
	p       probability
	see Hosking and Wallis, Technometrics, 1987
	"""
	# *** see 0001
	if np.abs(k)<1E-52:
		p = np.exp(-x/a)
	else:
		# original performed check of x<a/k
		# after computation
		# ultimatly correct result was used but only 
		# After a value error for raising a negative to a
		# a fractional power
		# 20121207 RAT
		p = np.zeros(len(x))
		p[(1.0-k*x/a)>=0] = (1.0-k*x/a)**(1.0/k)
		

	return p

def gpdcdf1(x,a,k):
	"""1 - CDF of the generalized pareto distribution, for scalar value of x
	x       exceedances
	k       shape parameter
	a       scale parameter
	return
	p       probability
	see Hosking and Wallis, Technometrics, 1987
	"""
	# added 20111205 RAT
	# *** 0001 this bit of code is redundant to gpdcdf
	# in the matlab code a scalar can be treated as a vector
	# so the code can run on either, and above the cdf is 
	# called for vector and then a scalar,
	# under certain conditions the scalar will fail
	# I wrote this as a lame work around 
	# should adapt gpdcdf to deal with scalars
	if np.abs(k)<1E-52:
		p = np.exp(-x/a)
	else:
		# original performed check of x<a/k
		# after computation
		# ultimatly correct result was used but only 
		# After a value error for raising a negative to a
		# a fractional power
		# 20121207 RAT
		p = 0.0
		if (1.0-k*x/a)>=0: p = (1.0-k*x/a)**(1.0/k)

	return p

	

def gpdgoft(p,k):
	"""Goodness of fit test for the generalized pareto distribution (gpd)
	P-value of the null hypothesis that the data comes from (or can be modeled with) the fitted gpd.
	Small p-values indicate a bad fit

	p	cdf values for data fitted to the gpd (from gpcdf)
	k	estimated shape parameter of the gpd (from gpfit)
	return
	Pad	P-value using Anderson-Darling statistic (this gives
        	more weight to observations in the tail of the distribution)
 
	Goodness-of-Fit Tests for the Generalized Pareto Distribution
 	Author(s): V. Choulakian and M. A. Stephens
 	Source: Technometrics, Vol. 43, No. 4 (Nov., 2001), pp. 478-484
 	Published by: American Statistical Association and American Society for Quality
 	Stable URL: http://www.jstor.org/stable/1270819
	"""
	p = np.sort(p)
	n = len(p)
	i = np.arange(1.0,n+1.0,1.0)
	# cramer-von mises statistic
	#W2 = np.sum((p-((2.0*i-1)/(2.0*n)))**2)+1.0/(12.0*n)
	# Anderson Darling statistic
	log1 = p
	log1[log1<1E-21] = 1E-21
	log2 = 1.0-p[::-1]
	log2[log2<1E-21] = 1E-21
	A2 = -n - np.dot(((1.0/n)*(2.0*i-1)),(np.log(log1)+np.log(log2)).T)
			
	k = max(.5,k)
	# extrapolate when necassary 
	#still working on this
	#f1 = _extrap1d(interp.interp1d(_ktable,_A2table.T))
	f1 = interp.interp1d(_ktable,_A2table.T)
	try:
		tmp = f1(k)
		# # extrapolate when necassary 
		# still working on this
		#f2 = _extrap1d(interp.interp1d(tmp,_ptable.T))
		f2 = interp.interp1d(tmp,_ptable.T)

		Pad = f2(A2)
	except ValueError:
		Pad = 0

	Pad = max(min(Pad,1),0)
	return Pad

def gpdPval(x0,parmhat):
	""" Computing P-value estimates for fitted data
	x0	test statistic compensated by exceedances threshold t
	parmhat	estimated shape and scale parameter
	Phat	estimated P-value
	"""
	Phat = gpdcdf1(x0,parmhat[0],parmhat[1])
	return Phat

def _extrap1d(interpolator):
	# returns an extrapolation when needed
	xs = interpolator.x
	ys = interpolator.y

	def pointwise(x):
		if x < xs[0]:
        		return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
		elif x > xs[-1]:
        		return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
		else:
        		return interpolator(x)
	def ufunclike(xs):
		# some problem in here ????
		return np.array(map(pointwise, np.array(xs)))

	return ufunclike