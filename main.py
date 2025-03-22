from kfp import dsl

# convert this into a pipeline component, 
# Run this in Python 3.12 environment
@dsl.component(base_image="python:3.12")
def power(base: int, exponent: int) -> int:
   result = 1
   for i in range(exponent):
      result = result * base 
   return result 


@dsl.component(base_image="python:3.12", packages_to_install=[ 'numpy', 'pandas' ])
def rand_num(low: int, high: int) -> int:
   # import must be inside the function
   import random   
   import pandas as pd
   import numpy as np
   return random.randint(low, high)

# Create a pipeline
@dsl.pipeline(name='random power', display_name='Contrive ML Pipeline')
def run_power_pipeline(init_low: int, init_high: int) -> int:
   # You must call the component with keyword arguments
   # CANNOT use positional arguments
   #_base = rand_num(init_low, init_high)
   _base = rand_num(high=init_high, low=init_low)
   _exp = rand_num(high=init_high, low=init_low)

   # Get the value of _base and _exp thru the .output 
   _result = power(base=_base.output, exponent=_exp.output)

   # return the result from the pipeline
   return _result.output
