OPENCV = `pkg-config opencv --cflags --libs`

run: main
	./main

main: main.cpp
	g++ main.cpp -o main  ${OPENCV}

