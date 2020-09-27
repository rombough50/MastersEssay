#Convert Strace To Integer [CoSTI]
#Writting by Jeffrey C. Rombough
#20190818 
#Used for data cleaning and processing of straces #to convert them into an integer form.

import os
import sys
import python_system_table_list as t


#Renames the file extension to csv
#Takes the individual trace files and merges them into one file
oldpath = "C:\\mx\\Convert Strace To Integer\\vx"
for file in os.listdir(oldpath):
    fullpath = oldpath + "\\" + file
    oldfile = open(fullpath,'r',encoding = 'utf-8')
    newpath = "C:\\mx\\Convert Strace To Integer\\vx Results\\" + file + ".csv"
    newfile  = open(newpath,'w',encoding = 'utf-8')
    for line in oldfile:
        justfirststring = line.split("(")[0]
        newfile.write(justfirststring)
        newfile.write("\n")


#Removes the arguments associated with the strace output
column1 = 1
colList = [ ]
csvpath = "C:\\mx\\Convert Strace To Integer\\vx Results"
for file in os.listdir(csvpath):
        fullpath = csvpath + "\\" + file
        csvhandle = open(fullpath,'r',encoding = 'utf-8')
        integerpath = "C:\\mx\\Convert Strace To Integer\\vx Integer Results\\" + file + ".csv"
        integerfile = open(integerpath,'w',encoding = 'utf-8')
        for line in csvhandle:
                for  thing in t.systemtable:
                        item = ','.join(thing)
                        sysnumber = item.split(',')[0]
                        syscallx = item.split(',')[1]
                        syscallx = syscallx.replace('"', '')
                        syscallx = syscallx + "\n"
                        if (line == syscallx):
                                integerfile.write(sysnumber + ",")
        integerfile.close()


#Where is the part which find the name of the system call and replaces it with the INT?


## Merge the CSV files into one CSV file
path_to_integer_csv = "C:\\mx\\Convert Strace To Integer\\vx Integer Results\\"
path_to_single_csv_file = "C:\\mx\Convert Strace To Integer\\VX_results.csv"
result_file = open(path_to_single_csv_file, "w", encoding = 'utf-8')
for file in os.listdir(path_to_integer_csv):
        full_integer_file_path = path_to_integer_csv + "\\" + file
        file_content = open(full_integer_file_path,"r",encoding = "utf-8")
        for line in file_content:
                result_file.write(line + "\n")

result_file.close()

