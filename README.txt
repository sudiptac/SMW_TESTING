
Download and install different implementations:
===============================================

Download all the code from this repository and follow the installation instruction 
as mentioned in the respective websites below: 

1. SSW library: 

First need to install the ssw library implementation to your system.
https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library

2. Jaligner:

First need to install the jaligner implementation to your system.
http://jaligner.sourceforge.net/

3. Diagonalsw:

First need to install the jaligner implementation to your system.
http://diagonalsw.sourceforge.net/#introduction

Changes in Implementations:
============================

1. In the case of diagonalsw implementation, change maximum length of the sequence from 2000 
	 to 10 million in the source code.


Running tests: 
==============

autotest/run_test is a bash script that run some random tests once you have installed all the 
software. The script will also provide the idea on how to run the different implementation. 
If you succesfully run the test script, it will produce a file "comparison.txt" in the "autotest" 
directory. Look out for the specific comments in the script. Be careful with all the parameters, 
as depending on the parameters, the behaviour of the respective implementation will vary. In order 
to compare different implementations, we must keep the parameters consistent across all implementations. 

