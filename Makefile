# Makefile

CXX = g++
CXXFLAGS = -Wall -g

SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)
	EXEC = transformer

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS) -lz3 `pkg-config --cflags --libs ginac`

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(EXEC)