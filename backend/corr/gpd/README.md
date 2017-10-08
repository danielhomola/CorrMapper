gpdPerm - genral pareto dist. for permutation testing 
=====================================================

Python module to improve pvalue estimation from a null distribution that is 
approximated by permutation testing. Uses a genralized pereto distribution 
to approximate the tail of the null distribution, as described in 
Knijnenburg 2009.  Authors publicly provided the original analysis as a set of 
matlab scripts.  

To run the full procedure given a test statistic and a 
vector of permutations call the method "est".

Knijnenburg 2009 -  Knijnenburg et al., Bioinformatics, Vol. 25 ISMB 2009, pages i161-i168


The current version is limited to the maximum lilklyhood method for preforming the fit.


Original code by: Theo Knijnenburg, Institute for Systems Biology, Jan 6 2009
Tranfered from matlab to python by: Ryan Tasseff, Institute for Systems Biology, Dec 2011

license
---------------------


Original Warranty Disclaimer and Copyright Notice attached to the matlab scripts:
 
Copyright (C) 2003-2010 Institute for Systems Biology, Seattle, Washington, USA.
 
The Institute for Systems Biology and the authors make no representation about the suitability or accuracy of this software for any purpose, and makes no warranties, either express or implied, including merchantability and fitness for a particular purpose or that the use of this software will not infringe any third party patents, copyrights, trademarks, or other rights. The software is provided "as is". The Institute for Systems Biology and the authors disclaim any liability stemming from the use of this software. This software is provided to enhance knowledge and encourage progress in the scientific community. 
 
This is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
