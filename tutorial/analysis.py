import sys
import csv
import matplotlib.pyplot as plt

# if len(sys.argv) < 4:
#     print("Invalid arguments")
#     exit()
files=[f"log{x}.csv" for x in sys.argv[1:]]


i=1
for file_name in files:
    x = []
    y = []
    with open(file_name,'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=",")
        for row in lines[5:]:
            x.append(int(row[0]))
            y.append(int(row[1]))
    label =""
    if i==1 :
        label = "P1"
    elif i==2:
        label = "P1 with competition"
    else :
        label = "P2"
        x=[ val+5 for val in x]
    i+=1
    plt.plot(x,y,label=label)
  
# plt.plot(x, y, color = 'g', linestyle = 'dashed',
#          marker = 'o',label = "Weather Data")
# plt.plot(x,y)
  
plt.xticks(rotation = 25)
plt.xlabel('Time')
plt.ylabel('Throughput')
plt.title('Performance report', fontsize = 20)
plt.grid()
plt.legend()
# plt.show()
plt.savefig("result.png")
