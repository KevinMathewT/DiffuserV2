#!/bin/bash
set -e  # Exit on error

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Root directory setup
echo -e "${GREEN}Setting up diffuser environment...${NC}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MUJOCO_DIR="${HOME}/.mujoco"
CONDA_ENV_NAME="diffuser"

# Step 1: Create conda environment
echo -e "${GREEN}Creating conda environment...${NC}"
conda env create -f environment.yml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

# Step 2: Download and set up MuJoCo 2.0.0
echo -e "${GREEN}Setting up MuJoCo 2.0.0...${NC}"
mkdir -p ${MUJOCO_DIR}

# Check if MuJoCo already exists
if [ ! -d "${MUJOCO_DIR}/mujoco200" ]; then
    echo -e "${YELLOW}Downloading MuJoCo 2.0.0...${NC}"
    wget https://www.roboti.us/download/mujoco200_linux.zip -O ${MUJOCO_DIR}/mujoco200_linux.zip
    unzip ${MUJOCO_DIR}/mujoco200_linux.zip -d ${MUJOCO_DIR}/
    mv ${MUJOCO_DIR}/mujoco200_linux ${MUJOCO_DIR}/mujoco200
    echo -e "${GREEN}MuJoCo 2.0.0 installed successfully.${NC}"
else
    echo -e "${YELLOW}MuJoCo 2.0.0 already installed, skipping...${NC}"
fi

# Copy MuJoCo license key
if [ -f "${SCRIPT_DIR}/extras/mjkey.txt" ]; then
    cp ${SCRIPT_DIR}/extras/mjkey.txt ${MUJOCO_DIR}/mjkey.txt
    echo -e "${GREEN}MuJoCo license key copied successfully.${NC}"
else
    echo -e "${RED}ERROR: MuJoCo license key not found at ${SCRIPT_DIR}/extras/mjkey.txt${NC}"
    echo -e "${RED}Please place your MuJoCo license key at this location and run the script again.${NC}"
    exit 1
fi

# Step 3: Set up environment variables for the current session
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_DIR}/mujoco200/bin
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_DIR}/mujoco200
export MUJOCO_PY_MJKEY_PATH=${MUJOCO_DIR}/mjkey.txt
export MUJOCO_GL=egl  # This was the key to making it work
export D4RL_SUPPRESS_IMPORT_ERROR=1

# Step 4: Create a conda activation script
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
cat > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh << EOF
#!/bin/bash
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${MUJOCO_DIR}/mujoco200/bin
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_DIR}/mujoco200
export MUJOCO_PY_MJKEY_PATH=${MUJOCO_DIR}/mjkey.txt
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF
chmod +x ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

# Step 5: Verify the installation
echo -e "${GREEN}Verifying MuJoCo-py installation...${NC}"
python -c "import mujoco_py; print('MuJoCo-py successfully imported')"

echo -e "${GREEN}Setup complete! You can now run:${NC}"
echo -e "${YELLOW}python -m scripts.train --config config.maze2d --dataset maze2d-large-v1${NC}"
echo -e "${GREEN}To activate the environment in the future, run:${NC}"
echo -e "${YELLOW}conda activate ${CONDA_ENV_NAME}${NC}"