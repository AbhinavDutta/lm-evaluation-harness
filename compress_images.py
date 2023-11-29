from PIL import Image
import os
import fnmatch
'''
DIR='./hostvm/artifacts/quantization/'
for file in os.listdir(DIR):
    if fnmatch.fnmatch(file, '*.png'):
        foo = Image.open(DIR+file)
        foo = foo.convert("P", palette=Image.ADAPTIVE, colors=64)
        foo.save(DIR+file,quality=75,optimize=True)
'''
foo=Image.open('/lm-evaluation-harness/hostvm/artifacts/batch/pretrainedmetallamaLlama27bchathffewshot0hendrycksTest-abstract_algebra_batch_size_1__a_likelihoods.txt_scatter_plot.png')
#foo = foo.resize((600,400),Image.LANCZOS)
print(foo.size)
foo = foo.convert("P", palette=Image.ADAPTIVE, colors=64)
foo.save('./hostvm/demo.png',optimize=True,quality=1)