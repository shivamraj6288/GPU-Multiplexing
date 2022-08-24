import sys
import csv
import matplotlib.pyplot as plt

# if len(sys.argv) < 4:
#     print("Invalid arguments")
#     exit()
files=[f"log{x}.csv" for x in sys.argv[1:]]

fig1 - plt.figure("without_restriction")
i=1
for file_name in files[0:3]:
    x = []
    y = []
    with open(file_name,'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=",")
        for row in lines:
            x.append(int(row[0]))
            y.append(int(row[1]))
    label =""
    if i==1 :
        label = "P1"
    elif i==2:
        label = "P1 with competition"
    else :
        label = "P2"
        x=[ val+10 for val in x]
    i+=1
    plt.plot(x[:-1],y[:-1],label=label)
  
# plt.plot(x, y, color = 'g', linestyle = 'dashed',
#          marker = 'o',label = "Weather Data")
# plt.plot(x,y)
  
# plt.xticks(rotation = 25)
plt.xlabel('Time')
plt.ylabel('Throughput')
# plt.title('Performance (without any restriction)', fontsize = 20)
plt.title('Performance (without any restriction)')
plt.grid()
plt.legend()
# plt.show()
plt.savefig("result_without_restriction.png")

fig2=plt.figure("result_with_restriction")
i=1
for file_name in files[3:]:
    x = []
    y = []
    with open(file_name,'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=",")
        for row in lines:
            x.append(int(row[0]))
            y.append(int(row[1]))
    label =""
    if i==1 :
        label = "P1"
    elif i==2:
        label = "P1 with competition"
    else :
        label = "P2"
        x=[ val+10 for val in x]
    i+=1
    plt.plot(x[:-1],y[:-1],label=label)
  
# plt.plot(x, y, color = 'g', linestyle = 'dashed',
#          marker = 'o',label = "Weather Data")
# plt.plot(x,y)
  
# plt.xticks(rotation = 25)
plt.xlabel('Time')
plt.ylabel('Throughput')
# plt.title('Performance (without any restriction)', fontsize = 20)
plt.title('Performance (with restriction)')
plt.grid()
plt.legend()
# plt.show()
plt.savefig(f"result_with_restriction.png")

fig3=plt.fig("Comparing independent performance")
with open(files[0],'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=",")
    for row in lines:
        x.append(int(row[0]))
        y.append(int(row[1]))
    label = "P1 without restriction"
    plt.plot(x[:-1],y[:-1],label=label)
with open(files[3],'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=",")
    for row in lines:
        x.append(int(row[0]))
        y.append(int(row[1]))
    label = "P1 with restriction"
    plt.plot(x[:-1],y[:-1],label=label)

plt.xlabel('Time')
plt.ylabel('Throughput')
plt.title('Performance')
plt.grid()
plt.legend()
plt.savefig(f"result_comparison.png")
