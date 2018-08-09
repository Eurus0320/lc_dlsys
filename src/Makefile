TARGET = src
SRCS = main.cpp

OBJS = $(SRCS:.c=.o)

$(TARGET):
	g++ -std=c++11 -shared -O3 -DNDEBUG -fPIC -o main.so main.cpp -lopenblas -m64
#	g++ -std=c++11 -shared -O3 -DNDEBUG -fPIC -o main.dll main.cpp -lopenblas -m64
