# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vladislave/coding/paralelka/task_1_sinus/with_cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vladislave/coding/paralelka/task_1_sinus/with_cmake

# Include any dependencies generated for this target.
include CMakeFiles/task1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/task1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/task1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/task1.dir/flags.make

CMakeFiles/task1.dir/lab1.cpp.o: CMakeFiles/task1.dir/flags.make
CMakeFiles/task1.dir/lab1.cpp.o: lab1.cpp
CMakeFiles/task1.dir/lab1.cpp.o: CMakeFiles/task1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vladislave/coding/paralelka/task_1_sinus/with_cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/task1.dir/lab1.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/task1.dir/lab1.cpp.o -MF CMakeFiles/task1.dir/lab1.cpp.o.d -o CMakeFiles/task1.dir/lab1.cpp.o -c /home/vladislave/coding/paralelka/task_1_sinus/with_cmake/lab1.cpp

CMakeFiles/task1.dir/lab1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task1.dir/lab1.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vladislave/coding/paralelka/task_1_sinus/with_cmake/lab1.cpp > CMakeFiles/task1.dir/lab1.cpp.i

CMakeFiles/task1.dir/lab1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task1.dir/lab1.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vladislave/coding/paralelka/task_1_sinus/with_cmake/lab1.cpp -o CMakeFiles/task1.dir/lab1.cpp.s

# Object files for target task1
task1_OBJECTS = \
"CMakeFiles/task1.dir/lab1.cpp.o"

# External object files for target task1
task1_EXTERNAL_OBJECTS =

task1: CMakeFiles/task1.dir/lab1.cpp.o
task1: CMakeFiles/task1.dir/build.make
task1: CMakeFiles/task1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vladislave/coding/paralelka/task_1_sinus/with_cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable task1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/task1.dir/build: task1
.PHONY : CMakeFiles/task1.dir/build

CMakeFiles/task1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/task1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/task1.dir/clean

CMakeFiles/task1.dir/depend:
	cd /home/vladislave/coding/paralelka/task_1_sinus/with_cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vladislave/coding/paralelka/task_1_sinus/with_cmake /home/vladislave/coding/paralelka/task_1_sinus/with_cmake /home/vladislave/coding/paralelka/task_1_sinus/with_cmake /home/vladislave/coding/paralelka/task_1_sinus/with_cmake /home/vladislave/coding/paralelka/task_1_sinus/with_cmake/CMakeFiles/task1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/task1.dir/depend

