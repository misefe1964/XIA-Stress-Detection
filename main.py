import os
import shutil
import csv

outputFile = open("preprocesseddata.csv")

os.chdir(os.path.join(os.getcwd(), 'Stress_dataset'))

# iterar sobre conteúdos da pasta os.getcwd (participantes)

# iterar sobre conteúdos da pasta de cada participante (eventos)

# em cada pasta de eventos

# pegar i-ésimo elemento de cada csv e colocar na 
# j-ésima coluna do pre-processed output file 