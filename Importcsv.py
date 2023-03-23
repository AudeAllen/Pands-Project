# Import csv file
import csv

# Open csv file

# Reference https://realpython.com/lessons/reading-csvs-pythons-csv-module/

with open ('iris_csv.csv') as iris_csv_file:    
        csv_reader  = csv.DictReader(iris_csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'column names are {", ".join(row)}')
                line_count += 1
            
            print(f'\t The sepal length is {row ["sepallength"]} and the sepal width is {row["sepalwidth"]}. The petal length is {row ["petallength"]}. Lastly the  petal width is {row ["petalwidth"]}')
            line_count +=1
            print(f'Processed  {line_count} lines.')          