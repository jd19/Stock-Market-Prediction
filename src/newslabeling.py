#from pandas import DataFrame, read_csv
import pandas as pd
from datetime import datetime
from dateutil import parser
import csv


aapldata=pd.read_csv("/home/soundarya/Downloads/Data/V.csv");
aapltimes=aapldata.iloc[:,0]
openvalues=aapldata.iloc[:,1]
closevalues=aapldata.iloc[:,4]

#Timestamps - Timestamps for news
#aapltimes - Timestamps for aapltimes(Financial data)

newsdata=pd.read_csv("/home/soundarya/Downloads/News/visa.csv");
timestamps=newsdata.iloc[:,0]
#print "length",len(timestamps)
label=[-1]*(len(timestamps))

month=[31,28,31,30,31,30,31,31,30,31,30,31]
monthleap=[31,29,31,30,31,30,31,31,30,31,30,31]

        

for i,time in enumerate(timestamps):
    print(i)
    name=parser.parse(time).strftime("%A")
    time=datetime.strptime(time, "%Y-%m-%dT%H:%M")
    Day=int(datetime.strftime(time,"%d"))
    Month=int(datetime.strftime(time,"%m"))
    Year=int(datetime.strftime(time,"%Y"))
    Hour=int(datetime.strftime(time,"%H"))
    if (name=="Friday" and ((Hour>=11) or (0<=Hour<=3))):
	if ((int(Year)%4)!=0):
        	if (((month[Month-1])-2)<=(Day)<=(month[Month-1])):
            		Day=Day%(month[Month-1]-3)
            		value=Month
            		Month=(Month%12)+1
            		if ((value-Month)==11):
                		Year=Year+1
            		Hour=5
		              
        	else:
            	    	Day=Day+3
            	    	Hour=5
	else:
		if (((monthleap[Month-1])-2)<=(Day)<=(monthleap[Month-1])):
            		Day=Day%(monthleap[Month-1]-3)
            		value=Month
            		#Stored the Month's value on value to increment year in future if there was a change from 12 to 1 (Month)
            		Month=(Month%12)+1
            		if ((value-Month)==11):
                		Year=Year+1
            		Hour=5
            	else:
			Day=Day+3
			Hour=5
            

    if name=="Saturday":
	if ((int(Year)%4)!=0):
        	if (((month[Month-1])-1)<=(Day)<=(month[Month-1])):
            		Day=Day%(month[Month-1]-2)
            		value=Month
            		#Stored the Month's value on value to increment year in future if there was a change from 12 to 1 (Month)
            		Month=(Month%12)+1
            		if ((value-Month)==11):
            		    Year=Year+1
        	
            		Hour=5
            
        	else:
            		Day=Day+2
            		Hour=5
	else:
		if (((monthleap[Month-1])-1)<=(Day)<=(monthleap[Month-1])):
            		Day=Day%(monthleap[Month-1]-2)
            		value=Month
            		#Stored the Month's value on value to increment year in future if there was a change from 12 to 1 (Month)
            		Month=(Month%12)+1
            		if ((value-Month)==11):
            		    Year=Year+1
        	
            		Hour=5
            
        	else:
            		Day=Day+2
            		Hour=5
    

    if name=="Sunday":
	if ((int(Year)%4)!=0):
        	if month[Month-1]==Day:
            		Day=Day%(month[Month-1]-1)
            		value=Month
            		#Stored the Month's value on value to increment year in future if there was a change from 12 to 1 (Month)
            		Month=(Month%12)+1
           		if ((value-Month)==11):
                		Year=Year+1
            
            		Hour=5
        
        	else:
            		Day=Day+1
            		Hour=5

	else:
		if monthleap[Month-1]==Day:
            		Day=Day%(monthleap[Month-1]-1)
            		value=Month
            		#Stored the Month's value on value to increment year in future if there was a change from 12 to 1 (Month)
            		Month=(Month%12)+1
           		if ((value-Month)==11):
                		Year=Year+1
            
            		Hour=5
        
        	else:
            		Day=Day+1
            		Hour=5
            


#If it is a sunday or a saturday, it won't go into this loop!

    if ((Hour>=11) or (0<=Hour<=3)):
            Hour=5
            if ((int(Year)%4)!=0):
                
                if (Day==month[int(Month-1)]):
                    Day=1
                    Month=((int(Month))%12)+1
        

                else:
                    Day=int(Day)+1
                    
                
                
            

            else:
                if (Day==monthleap[int(Month-1)]):
                    Day=1
                    Month=((int(Month))%12)+1
            
        

                else:
                    Day=int(Day)+1
                    
    
              
    for j,ftime in enumerate(aapltimes):
        ftime = datetime.strptime(ftime, "%Y-%m-%d\t%H:%M:%S")
        ftimeyear=datetime.strftime(ftime,"%Y")
        if (int(ftimeyear)==Year):
            ftimehour=datetime.strftime(ftime,"%H")
            if(int(ftimehour)==(Hour+4)):
                ftimemonth=datetime.strftime(ftime,"%m")
                if(int(ftimemonth)==Month):
                    ftimeday=datetime.strftime(ftime,"%d")
                    if (int(ftimeday)==Day):
                        print(closevalues[j],openvalues[j])
                        if (closevalues[j]>openvalues[j]):
                            label[i]=1
                            break
                        else:
                            label[i]=-1
                            break
    
                    
print(label)     
print (len(label))              
print (len(timestamps))
f=open('labelsforv.csv', 'a')
Writer = csv.writer(f, delimiter=' ',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
for i in label:
    Writer.writerow([i])
f.close()

        
            
                


            
            
            


                
        

            
            
            
