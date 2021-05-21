###MolPred README

- _MolPred_ is a tool designed to predict the value of the column density (_log(N)_) and the exitation temperature 
(T<sub>ex</sub>). It works by performing a regression with several neural networks trained from examples coming from MADCUBA. 

### Installation Instructions (Windows)

- Install virtualenv
    `python -m pip install --user virtualenv`
- Create virtual environment
    `python -m venv molpred_env`
- Activate virtual environment
    `.\molpred_env\Scripts\activate`
- Unpack MolPred.rar
- Navigate to your decompressed MolPred directory    
- Check that your molpred_env python appears first in the list
    `where python`
- Install the package requirements list
    `pip install -r requirements-minimal.txt`
####Notes
- Tensorflow package needs CUDA to properly use the NVIDIA GPU cards, without this, you might find warnings in the code
execution, you can find instructions here: 
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

### Installation Instructions (Linux)

- Install virtualenv
    `python3.8 -m pip install --user virtualenv`
- Create virtual environment (python3.8 required)
    `python3.8 -m venv molpred_env`
- Activate virtual environment
    `source molpred_env/bin/activate`
- Unpack MolPred.rar
- Navigate to your decompressed MolPred directory    
- Check that your molpred_env python appears first in the list
    `which python`
- Install the package requirements list
    `pip3.8 install -r requirements-minimal.txt`
    
        
### Test MolPred
- Activate virtual environment as shown in the installation section.
- Navigate to MolPred/src directory
- Launch MolPred with test spectrum
    `python MolPred.py --specfile ../example_input/ALCHEMI_ACA.data`
- When finishing, predictions should be left in directory "predictions"
- To finish using the virtual environment, deactivate it with
    `deactivate`

### Release Notes v1.0
- This is our first release of MolPred. It's still pretty basic, but it's our starting point
- MolPred contains only the best models per molecule from our study, for now. 
In next updates, we will include scripts to allow you to train your own neural network models and used them
with MolPred.
- An overview of Molpred and what it does, can be read in our paper (Currently submitted to Experimental Astronomy).
- More technical details, will be available in the Thesis (in preparation).
    
     
