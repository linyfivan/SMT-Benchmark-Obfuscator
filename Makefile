# Makefile

CXX = g++
CXXFLAGS = -Wall -g

SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)
	EXEC = transformer

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS) -lz3

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -lz3

clean:
	rm -f $(OBJS) $(EXEC)