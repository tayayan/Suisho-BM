# Suisho-BM - Makefile (YaneuraOu USER_ENGINE edition)
#
#   make                      # g++, TARGET_CPU=AVX2
#   make TARGET_CPU=ZEN2      # AMD Zen 2 (BMI2 is slow there)
#   make TARGET_CPU=SSE41     # older CPUs without AVX2
#   make COMPILER=clang++
#
# On Windows build with MSYS2 MinGW64 (see ../build.ps1).

TARGET_CPU = AVX2
COMPILER   = g++
OBJDIR     = ../obj
TARGETDIR  = ..

CPPFLAGS = -std=c++17 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive \
           -DNDEBUG -DUNICODE -DNO_EXCEPTIONS -DUSER_ENGINE -DTARGET_CPU=\"$(TARGET_CPU)\"
LDFLAGS  = -lpthread -Wl,-s -flto

ifeq ($(TARGET_CPU),ZEN2)
	CPPFLAGS += -DUSE_AVX2 -mbmi -mno-bmi2 -mavx2 -march=znver2
else ifeq ($(TARGET_CPU),ZEN3)
	CPPFLAGS += -DUSE_AVX2 -DUSE_BMI2 -mbmi -mbmi2 -mavx2 -march=znver3
else ifeq ($(TARGET_CPU),AVX512)
	CPPFLAGS += -DUSE_AVX512 -DUSE_BMI2 -march=skylake-avx512
else ifeq ($(TARGET_CPU),AVX2)
	CPPFLAGS += -DUSE_AVX2 -DUSE_BMI2 -mbmi -mbmi2 -mavx2 -march=corei7-avx
else ifeq ($(TARGET_CPU),SSE42)
	CPPFLAGS += -DUSE_SSE42 -msse4.2 -march=corei7
else ifeq ($(TARGET_CPU),SSE41)
	CPPFLAGS += -DUSE_SSE41 -msse4.1 -march=core2
else ifeq ($(TARGET_CPU),SSE2)
	CPPFLAGS += -DUSE_SSE2 -msse2 -march=core2
endif

ifneq (,$(findstring g++,$(COMPILER)))
	CPPFLAGS += -fno-asynchronous-unwind-tables
endif

# Windows: cmd/PowerShell set OS=Windows_NT; an MSYS2 login shell does not,
# so check uname too. Link statically so that no MinGW DLLs are needed.
UNAME := $(shell uname -s 2>/dev/null)
ifneq (,$(filter Windows_NT,$(OS))$(findstring MINGW,$(UNAME))$(findstring MSYS,$(UNAME)))
	LDFLAGS += -static -Wl,--stack,25000000
	TARGET = $(TARGETDIR)/Suisho-BM.exe
else
	CPPFLAGS += -D_LINUX
	TARGET = $(TARGETDIR)/Suisho-BM
endif

SOURCES = \
	main.cpp \
	types.cpp \
	bitboard.cpp \
	misc.cpp \
	movegen.cpp \
	position.cpp \
	usi.cpp \
	usi_option.cpp \
	thread.cpp \
	tt.cpp \
	movepick.cpp \
	timeman.cpp \
	extra/bitop.cpp \
	extra/long_effect.cpp \
	extra/sfen_packer.cpp \
	extra/super_sort.cpp \
	mate/mate.cpp \
	mate/mate1ply_without_effect.cpp \
	mate/mate1ply_with_effect.cpp \
	eval/evaluate.cpp \
	eval/evaluate_bona_piece.cpp \
	eval/material/evaluate_material.cpp \
	testcmd/benchmark.cpp \
	engine/user-engine/solver.cpp \
	engine/user-engine/user-search.cpp

OBJECTS = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
DEPENDS = $(OBJECTS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) -o $@ $^ $(LDFLAGS) $(CPPFLAGS)

$(OBJDIR)/%.o: %.cpp
	@[ -d $(dir $@) ] || mkdir -p $(dir $@)
	$(COMPILER) $(CPPFLAGS) -flto -o $@ -c $<

clean:
	rm -rf $(OBJDIR) $(TARGET)

-include $(DEPENDS)
