import json

file = open('trace_4bit_128batch.json')
kernels=[]
data = json.load(file)

for i in data['traceEvents']:
    kernels.append(i['name'])
    
unique_kernels = sorted(set(kernels))
for u in unique_kernels:
    if 'gemm' in u:
        print(u)
# Closing file
file.close()
