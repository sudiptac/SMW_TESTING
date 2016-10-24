import random

#Create insert, delete, swap and operate functions.
def insert(d1,s):
	position=random.randint(0,len(d1))
	index=random.randint(0,22)
	d1.insert(position,s[index])
def delete(d1,s):
	position=random.randint(0,len(d1))
	del d1[position]
def swap(d1,s):
	position=random.randint(0,len(d1))
	index=random.randint(0,22)
	d1[position]=s[index]
def operate(d1,s):
	index=random.randint(0,2)
	if index == 0:
		insert(d1,s)
	if index == 1:
		delete(d1,s)
	if index == 2:
		swap(d1,s)
	return d1

def smith(d1,data1,times):
	file = open(d1, 'r')
	header = file.readline()
	datalist = []
	for line in file:
		linesplit = list(line)
		linesplit.pop()
		datalist.append(linesplit)
	datalist = sum(datalist, [])

	s = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X']
#	s1 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X']

	for i in range(times):
		datalist=operate(datalist,s)
	#mutatedlist=to_matrix(datalist, 70)

	#Writing a file.
	fileout = open(data1, 'w')
	fileout.write(header)
	for i in datalist:
		fileout.write(i)

if __name__ == "__main__":
#play with the numbers to generate more and more mutated inputs
	for i in range(151,161,2):
		t=random.randint(0,200)
		smith('d1.fasta', 'data'+str(i)+'.fasta',t)
		smith('d2.fasta', 'data'+str(i+1)+'.fasta',t)





#Create insert, delete, swap and operate functions.
def insert(da1,s):
	position=random.randint(0,len(da1))
	index=random.randint(0,22)
	da1.insert(position,s[index])
def delete(da1,s):
	position=random.randint(0,len(da1))
	del da1[position]
def swap(da1,s):
	position=random.randint(0,len(da1))
	index=random.randint(0,22)
	da1[position]=s[index]
def operate(da1,s):
	index=random.randint(0,2)
	if index == 0:
		insert(da1,s)
	if index == 1:
		delete(da1,s)
	if index == 2:
		swap(da1,s)
	return da1

def smith(da1,dataa1,times):
	file = open(da1, 'r')
	header = file.readline()
	datalist = []
	for line in file:
		linesplit = list(line)
		linesplit.pop()
		datalist.append(linesplit)
	datalist = sum(datalist, [])

	s = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X']
#	s1 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','B','Z','X']

	for i in range(times):
		datalist=operate(datalist,s)
	#mutatedlist=to_matrix(datalist, 70)

	#Writing a file.
	fileout = open(dataa1, 'w')
	fileout.write(header)
	for i in datalist:
		fileout.write(i)

if __name__ == "__main__":
	for i in range(1,11,2):
		t=random.randint(0,200)
		smith('da1.fasta', 'dataa'+str(i)+'.fasta',t)
		smith('da2.fasta', 'dataa'+str(i+1)+'.fasta',t)
