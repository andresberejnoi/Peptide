import tensorflow as tf
import csv

f = open('traindata.csv','rb+')   #opening file in rb+ format
f2 = open('output.csv','wb+')    #file to append 0's and 1's based on protien value

try:

	reader = csv.reader(f) #creating reader instance
	writer = csv.writer(f2) #creating writer instance 
	rownum = 0;
	
	for row in reader:

		if rownum == 0:
		 	header = row
		else:
			colnum = 6	#protien column number 
			column = 0
			for col in row:
				column = column+1

				if column == 7:
					print '%-8s: %s' %(header[colnum], col) #prints protien header
					result = col.startswith('Reverse') #holds the value if its starting with word 'reverse'
					print result #printing it just to make sure
					if result is False: 
						writer.writerow(row+['1'])
						#writer.writerow(row+['Similar'])	#appends 1 if the value is not starting with reverse
						print "1"			#printing it on screen 
						
						#print "Similar"

					else:
						writer.writerow(row+['0'])       #appends 0 if its starting with 0
						print "0"							
						#writer.writerow(row+['Not Similar'])
						#print "Not similar"
						 
						
						
				#colnum += 1
		rownum += 1
finally:
	f.close()
	f2.close()

#Training the model





