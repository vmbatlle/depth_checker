SH_PATH="`dirname \"$0\"`"
mkdir -p "${SH_PATH}/build"
cd "${SH_PATH}/build"
LIBTORCH_PATH="${SH_PATH}/../libtorch"
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" ..
cmake --build . --config Release