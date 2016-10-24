Basics:
=======

1. generate datasets by using generate_test_data.py python script, which is provided in the repository.

2. Install the implementation first and run the implementations by using commands, which 
is provided in the following:


Commands to run the different implementations:
===============================================

I feed the implementations with PROTEIN sequence as input which are mutations in the sequences
of both query and reference. I replaced, deleted and inserted the amino acids in both query and
reference sequences. Then I gave it as input to the all three implementations.

1. SSW Library:

First need to install the ssw library implementation to your system.
https://github.com/mengyao/Complete-Striped-Smith-Waterman-Library

Input of Implementation:
Feed the implementation with data1.fasta as query sequence, data2.fasta as reference sequence
and blosum62.mat as matrix.

Command to run:
ssw_test -a matrices/blosum62.mat ./data1.fasta ./data2.fasta

Output:
We get the output in the form of quality score for a given input query, reference sequence and
blosum matrix.

2. Jaligner:

First need to install the jaligner implementation to your system.
http://jaligner.sourceforge.net/

Input of Implementation:
Feed the implementation with data1.fasta as query sequence, data2.fasta as reference sequence
and blosum62.mat as matrix.

Command to run:
java -Xmx4096M -jar jaligner.jar data1.fasta data2.fasta BLOSUM62 10.0 0.5

Output:
We get the output in the form of quality score for a given input query, reference sequence and
blosum matrix.

3. Diagonalsw:

First need to install the jaligner implementation to your system.
http://diagonalsw.sourceforge.net/#introduction

Input of Implementation:
Feed the implementation with data1.fasta as query sequence, data2.fasta as reference sequence
and blosum62.mat as matrix.

Command to run:
./diagonalsw -q test/data1.fasta -d test/data2.fasta -s matrices/blosum62.mat

Output:
We get the output in the form of quality score for a given input query, reference sequence and
blosum matrix.


Changes in Implementations:
============================

1. In the case of diagonalsw implementation, I changed maximum length of the sequence from 2000 
	 to 10 million in the source code.

2.   In the case of Jaligner implementation, I need to run by below command.

         java -Xmx4096M -jar jaligner.jar data1.fasta data2.fasta BLOSUM62 10.0 0.5
