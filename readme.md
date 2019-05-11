# Skeleton Convolution Network(SCN) based on the Deep Graph Library(DGL)





In this repository, I am trying to implement simple skeleton based action recognition using Graph Convolution Network(GCN)[1] with the support of Deep Graph Library(DGL)[2]. 



-----



## About the database

I evaluate the code on the NTU RGB+D database[3]. The original skeleton data is saved as string in the `txt` files, which need to be parsed while loading data. To speed up the loading, I coded a script in `tools/ntu_read_skeleton.py` to transform the skeleton data into `numpy.narray`. Each sample is named following the rule:

```python
'S{:0>3}C{:0>3}P{:0>3}R{:0>3}A{:0>3}.skeleton.npy'.format(setup_id, camera_id, person_id, repeat_id, action_id)
```



You can generally load the sample and index the sample like:

```python
import numpy as np
sample = np.load(file_name).item()
mat_0 = sample['body0'] # first actor
mat_1 = sample['body1'] # second actor
```



There are two sample instances in the folder `data_samples/` for a better understand of the data format.



----



## Problems



- 4/24, 2019: The network cannot converge. The gradient of the GCN seems not change.
- 4/24, 2019: The network can converge now but still doesn't converge well. The GPU usage is low, maybe the problem of `DataLoader` design.
- 5/9, 2019: referring the design in STGCN[4] and have a better network and software framework designing now!



-----

# Reference

[1]. Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016. 

[2]. <https://github.com/dmlc/dgl>

[3]. <https://github.com/shahroudy/NTURGB-D>

[4]. https://github.com/yysijie/st-gcn

