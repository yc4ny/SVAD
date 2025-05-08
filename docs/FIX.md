# Package Fixes

After installing the Avatar environment, some package files need to be modified to fix compatibility issues. 
Apply the following changes:

**Issue** ❌ RuntimeError: Subtraction, the '-' operator, with a bool tensor is not supported.

File: **~/miniconda3/envs/svad/lib/python3.9/site-packages/torchgeometry/core/conversions.py**

Change: Lines #301-304 

Previous:
```bash
mask_c0 = mask_d2 * mask_d0_d1                    
mask_c1 = mask_d2 * (1 - mask_d0_d1)            
mask_c2 = (1 - mask_d2) * mask_d0_nd1           
mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)     
```

Update:
```bash
mask_c0 = mask_d2.float() * mask_d0_d1.float()              
mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())        
mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()       
mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float()) 
```
<br>
<br>

**Issue** ❌ cannot import name 'cached_download' from 'huggingface_hub' 
File: **~/miniconda3/envs/svad/lib/python3.9/site-packages/diffusers/utils/dynamic_modules_utils.py**

Change: Line #28

Previous: 
```bash
from huggingface_hub import cached_download, hf_hub_download, model_info
```

Update: 
```bash
from huggingface_hub import hf_hub_download, model_info
```
<br>
<br>

**Issue** ❌ No module named 'torchvision.transforms.functional_tensor'


File: **~/miniconda3/envs/svad/lib/python3.9/site-packages/basicsr/data/degradations.py**

Change: Line #8

Previous: 
```bash
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

Update: 
```bash
from torchvision.transforms.functional import rgb_to_grayscale
```
<br>
<br>

**Issue** ❌ AssertionError: MMCV==2.2.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.2.0.


File: **~/miniconda3/envs/svad/lib/python3.9/site-packages/mmdet/\_\_init\_\_.py**

Change: Line #8

Previous: 
```bash
mmcv_maximum_version = '2.2.0'
```

Update: 
```bash
mmcv_maximum_version = '2.2.1'
```