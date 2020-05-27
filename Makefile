ifeq ($(OS),Windows_NT)
	RM := del /s
	LDLIBS := -llibopencv_core349 -llibopencv_videoio349 
	LDLIBS += -llibopencv_imgcodecs349 -lopencv_highgui349
	LDLIBS += -llibopencv_imgproc349
else
	LDLIBS := $(shell pkg-config --cflags --libs opencv)
	LDLIBS += -pthread
endif

CPPFLAGS := "-IC:\\opencv-3.4.9\\build\\include"
CXXFLAGS := -std=c++14 -Wall -W -O3
LDFLAGS := "-LC:\\opencv-3.4.9\\x86\\lib" 

.PHONY: all clean

all: read_npy

read_npy: read_npy.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	$(RM) read_npy