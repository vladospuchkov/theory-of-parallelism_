INFO = -Minfo=all
LIBS = -lboost_program_options
CXX = g++

all:test
	
test (test.cpp):
	$(CXX) $(INFO) $(LIBS) -o $@ test.cpp

clean:all
	rm test