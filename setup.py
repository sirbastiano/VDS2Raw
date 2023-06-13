from setuptools import setup, find_packages
import os
import subprocess
import platform

def is_linux():
    print("Platform system: ",platform.system())
    return platform.system() == 'Linux'

if is_linux():
    print("The system is running on Linux.")
    system = 0
    
else:
    print("The system is not running on Linux.")
    system = 1

def install_dependencies():
    torch_dependencies = "https://download.pytorch.org/whl/torch_stable.html"

    if system == 0:
        subprocess.check_call(["pip", "install", "torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "-f", torch_dependencies])
    else:
        subprocess.check_call(["pip", "install", "torch==2.0.0", "torchvision==0.15.1", "-f", torch_dependencies])
        
    subprocess.check_call(["pip", "install", "SciencePlots"])
    subprocess.check_call(["pip", "install", "openmim"])
    subprocess.check_call(["mim", "install", "mmengine"])
    subprocess.check_call(["mim", "install", "mmcv>=2.0.0"])
    subprocess.check_call(["git", "clone", "https://github.com/open-mmlab/mmdetection.git"])
    os.chdir("./mmdetection")
    subprocess.check_call(["pip", "install", "-v", "-e", "."])
    os.chdir("..")
    subprocess.check_call(["conda", "install", "-c", "conda-forge", "tifffile"])

install_dependencies()