[![DOI](https://zenodo.org/badge/351828358.svg)](https://zenodo.org/doi/10.5281/zenodo.13761539)

# approximate_mle
Numerical experiments for approximate maxmimum likelihood estimation based on saddle point approximations. 

This repo includes code used for numerical experiments to test our proposed approximate maximum likelihood method in both MATLAB and Python as discussed the . The MATLAB code supports experiments using synethic data as described in section 6.1, and the Python code supports the housing data problems discussed discussed 


The following repo includes code used to test our proposed approximate likelihood method as described in our preprint, [Approximate maximum likelihood estimators for linear regression with design matrix uncertainty](https://arxiv.org/abs/2104.03307). The Matlab code supports synthetic data outlined in the numerical experiments and the Python code supports a more general framework which was used for a housing data example. 

# MATLAB code
There are several scripts written in Matlab that generate data output presented in the numerical experiments section. The following files generate output:
- `vary_column_count_combined.m`: varies the number of columns while holding the number of rows fixed in the design matrix (while remaining over determined)
- `vary_row_count_combined.m`: varies the the number of rows while holding the number of columns fixed in the design matrix (while remaining over determined)
- `fixed_row_and_column_count.m`: this script generates output that fixes the number of rows and columns used in the design matrix for different noise set ups. This is used to generate box-plots and histograms seen in Figure 4 in MPC version. 

The following functions are auxillary functions to generate corresponding plots:
- `plot_fixed_row_and_column_count.m`
- `plot_vary_by_cols_combined.m`
- `plot_vary_by_rows_combined.m`

The `functions` folder includes helper functions used to generate data such as Newton's method. 


## To run....

Navigate to the directory where the files are located then run the desired script. Corresponding plotting routines will execute at the bottom of each script. 


# Python Code
The Python code has the following requirements: 
```
- numpy
- pandas
- scipy 
- matplotlib
```

Ensure that the following directories are in your PYTHONPATH.
```
{$repo_home}/approximate_mle/
{$repo_home}/approximate_mle/python
```

Once properly set up, you can run `housing_prices.py` which will excecute the code and save a pdf file.
