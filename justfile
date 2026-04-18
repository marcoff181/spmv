
init:
    cmake -B build
    ln -sf build/compile_commands.json .

build:
    cmake --build build

run:
    ./build/spmv

clean:
    rm -rf build
    rm -f compile_commands.json
