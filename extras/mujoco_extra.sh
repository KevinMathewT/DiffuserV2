#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <netid>"
    exit 1
fi

NETID=$1

# -------- CONFIG --------
MJ_VERSION="mujoco200"
MJ_DIR="/scratch/${NETID}/${MJ_VERSION}"
MJ_ZIP="${MJ_VERSION}_linux.zip"
MJ_URL="https://www.roboti.us/download/${MJ_ZIP}"
MJ_LOCAL_ZIP="/scratch/${NETID}/${MJ_ZIP}"
MJ_SYMLINK="$HOME/.mujoco"
MJ_KEY_SRC="./extras/mjkey.txt"

# -------- CLEAN OLD INSTALLS --------
echo "[1/6] Cleaning previous installations..."
rm -rf $MJ_DIR $MJ_SYMLINK ~/.mujoco-py ~/.cache/mujoco_py

# -------- DOWNLOAD MUJOCO 2.0 --------
echo "[2/6] Downloading MuJoCo 2.0 to $MJ_DIR..."
wget -O $MJ_LOCAL_ZIP $MJ_URL
unzip $MJ_LOCAL_ZIP -d /scratch/${NETID}
mv /scratch/${NETID}/${MJ_VERSION}_linux $MJ_DIR
rm -f $MJ_LOCAL_ZIP

# -------- SET UP ~/.mujoco --------
echo "[3/6] Setting up ~/.mujoco..."
mkdir -p ~/.mujoco
ln -s $MJ_DIR ~/.mujoco/mujoco200
cp $MJ_KEY_SRC ~/.mujoco/

# -------- INSTALL mujoco-py HEADLESS --------
echo "[4/6] Installing mujoco-py with headless (osmesa) backend..."
pip uninstall -y mujoco-py || true
pip install git+https://github.com/openai/mujoco-py.git@master

# -------- SET ENV VARS --------
echo "[5/6] Exporting environment variables..."
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH=$MJ_DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MJ_DIR/bin

grep -qxF 'export MUJOCO_GL=osmesa' ~/.bashrc || echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc
grep -qxF "export MUJOCO_PY_MUJOCO_PATH=$MJ_DIR" ~/.bashrc || echo "export MUJOCO_PY_MUJOCO_PATH=$MJ_DIR" >> ~/.bashrc
grep -qxF "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MJ_DIR/bin" ~/.bashrc || echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$MJ_DIR/bin" >> ~/.bashrc

# -------- TEST BUILD --------
echo "[6/6] Verifying mujoco-py install..."
python -c "import mujoco_py; print('✅ mujoco-py import successful!')"

echo "✅ All done. You are ready to use mujoco-py in headless mode (CPU only)."
