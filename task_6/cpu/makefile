INFO = -Minfo=all
LIBS = -lboost_program_options
HOST = -acc=host
MULT = -acc=multicore
CXX = pgc++

all:serial multi
	

serial:
	$(CXX) $(HOST) $(INFO) $(LIBS) -o $@ cpu.cpp

multi:
	$(CXX) $(MULT) $(INFO) $(LIBS) -o $@ cpu.cpp

clean:all
	rm serial multi