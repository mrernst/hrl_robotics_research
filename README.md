# HRL Robotics Research
<p align="center">
  <img src="https://github.com/mrernst/hrl_robotics_research/blob/main/img/HIRO_PER_t5M.gif" width="640">

### Project Name LUNA

This is an active development and research repository for hierarchical reinforcement learning. It contains a complete reimplementation of the HIRO agent ("Data-Efficient Hierarchical Reinforcement Learning" - Ofir Nachum et al., 2018) and builds on it in various domains, i.e. Prioritized Experience Replay and Subgoal compression.
 
It is build in a way to incorporate OpenAI Gym environments and and Mujoco/PyBullet for robotic tasks.

### Flat agent
Standard TD3 Algorithm

### Hierarchical agent
HIRO

Baymax (Additional Subgoal Compression)

# Notes/ Acknowledgements
The launcher script is adapted from IAS TU Darmstadt -> https://gitlab.ias.informatik.tu-darmstadt.de/common/experiment_launcher
and simplyfies sending jobs to a slurm cluster.


# Running Mujoco as a Simulator for RL Tasks

## Mujoco on Apple Silicon

### Installation Guide
1) Make directories for mujoco-py and link Mujoco of the App Bundle
> mkdir -p $HOME/.mujoco/mujoco210
> ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include

2) Link library files
> mkdir -p $HOME/.mujoco/mujoco210/bin
> ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
> sudo ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/

3) Install needed graphics library via homebrew
> brew install glfw
> ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

4) Remove old installation
> rm -rf /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/mujoco_py

> which python
> exit

5) Add CC to your Bash/ZSH Variables
> export CC=/opt/homebrew/bin/gcc-12         # see https://github.com/openai/mujoco-py/issues/605
> pip install mujoco-py && python -c 'import mujoco_py'



## Mujoco on the FIAS Cluster

### Installation Guide
1) Setup a miniconda environment with python 3.9.X and install pytorch, gym, dependencies you need for development
> conda install numpy
> conda install pytorch
> pip install gym

2) Install mujoco 2.1.0 from OpenAI, add to default folder at .mujoco
> wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz --no-check-certificate
> tar -xvzf mujoco210-linux-x86_64.tar.gz
> mkdir ~/.mujoco
> mv mujoco210 ~/.mujoco/mujoco210

3) Install mujoco-py via pip (not executable yet)
> pip3 install mujoco-py


4) Use conda to install patchelf, menpo, osmesa because no root access at FIAS computers
> conda install patchelf
> conda install -c conda-forge menpo
>  conda install -c menpo osmesa 

5) Get source of libgcrypt <= 1.5.3! and compile yourself (https://www.gnupg.org/ftp/gcrypt/libgcrypt/)
> wget https://www.gnupg.org/ftp/gcrypt/libgcrypt/libgcrypt-1.5.3.tar.gz --no-check-certificate

6) Make a directory for custom libraries and move stuff there
> mkdir ~/opt
> mkdir ~/opt/lib
> mkdir ~/opt/lib/libgcrypt
> mv libgcrypt-1.5.3.tar.gz ~/opt/lib/
> cd ~/opt/lib
> tar -xvzf libgcrypt-1.5.3.tar.gz
> cd libgcrypt-1.5.3
> ./configure --prefix=/home/FIAS_USER_NAME/opt/lib/libgcrypt && make
> make install

7) You should append mujoco and prerequisites into the respective UNIX path variables because at first executing mujoco-py compiles it's C-language parts

> vim ~/.bashrc

and insert:

export LD_LIBRARY_PATH=/home/FIAS_USER_NAME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/FIAS_USER_NAME/opt/lib/libgcrypt/lib:$LD_LIBRARY_PATH
export PATH=/home/FIAS_USER_NAME/opt/lib/libgcrypt/bin:$PATH
export C_INCLUDE_PATH=/home/FIAS_USER_NAME/opt/lib/libgcrypt/include:$C_LIBRARY_PATH:$C_INCLUDE_PATH

8) logout and login, or restart your shell

9) Hopefully mujoco works, try
> python -c "import mujoco_py"
