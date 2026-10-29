import nbformat
from nbclient import NotebookClient
import os
import sys

print("Executing Automation EDA.ipynb to populate all cell outputs...")

with open('Automation EDA.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3', allow_errors=True)
client.execute()

with open('Automation EDA.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook execution finished! All cell outputs are saved in Automation EDA.ipynb.")
