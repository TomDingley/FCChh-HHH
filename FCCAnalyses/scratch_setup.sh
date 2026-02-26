rm -rf build install
mkdir build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make install -j
cd ..
source setup.sh