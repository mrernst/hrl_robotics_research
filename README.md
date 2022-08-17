# RL Robotics Research Playground
### Project Name LUNA

Machine learning playground using OpenAI Gym and Mujoco/PyBullet for robotic tasks.

### Flat agent
TD3 Algorithm

### Hierarchical agent


# Notes/ Acknowledgements
Launcher is adapted from IAS TU Darmstadt -> https://gitlab.ias.informatik.tu-darmstadt.de/common/experiment_launcher

# Mujoco on Apple Silicon

mkdir -p $HOME/.mujoco/mujoco210
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/Headers/ $HOME/.mujoco/mujoco210/include

mkdir -p $HOME/.mujoco/mujoco210/bin
ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib $HOME/.mujoco/mujoco210/bin/libmujoco210.dylib
sudo ln -sf /Applications/MuJoCo.app/Contents/Frameworks/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib /usr/local/lib/

# For M1 (arm64) mac users:
# brew install glfw
ln -sf /opt/homebrew/lib/libglfw.3.dylib $HOME/.mujoco/mujoco210/bin

# remove old installation
rm -rf /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/mujoco_py

# which python
# exit

export CC=/opt/homebrew/bin/gcc-12         # see https://github.com/openai/mujoco-py/issues/605
#pip install mujoco-py && python -c 'import mujoco_py'