OPENCV = `pkg-config opencv --cflags --libs`

main: *.cpp
	g++ *.cpp -o main  ${OPENCV}

debug: *.cpp
	g++ *.cpp -g -o main  ${OPENCV}

run: main
	./main

debugrun: debug
	cgdb main

clean:
	rm main
	rm *~

