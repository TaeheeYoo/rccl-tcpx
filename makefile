# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------
#  Layout
LIB_DIR := library
SRC_DIR := source
INC_DIR := include
OUT_DIR := output
CNF_DIR := config

# Configures
LIBFILE := $(CNF_DIR)/library.mk
RUNFILE := $(CNF_DIR)/run.mk
BLDFILE := $(CNF_DIR)/build.mk

#  Program
MKDIR := mkdir -p
WGET := wget
RM := rm -f
MV := mv
TEST := test
SORT := sort
GREP := grep
AWK := awk
PR := pr
SED := sed
LN := ln -s
CAT := cat
TOUCH := touch
CTAGS := ctags
CSCOPE := cscope -b
BEAR := bear

# Variables 
SOURCES :=
OBJECTS :=
INCLUDES :=
DEPENDENCIES :=

LIBRARIES := $(file < $(LIBFILE)) $(patsubst $(LIB_DIR)/%,%,$(foreach u,$(wildcard $(LIB_DIR)/*),$(wildcard $u/*)))

# Output
OUTPUT ?= program

CSCOPE_FILE_OUT := cscope.files
CSCOPE_DB_OUT := cscope.out

CTAGS_OUT := tags

COMPILE_DB_OUT := compile_commands.json

# Constants
SYNC_TIME := $(shell LC_ALL=C date)

# Internal
.DEFAULT_GOAL = help

-include $(BLDFILE)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
# $(call get-source-file,dir) -> source-list
get-source-file = $(wildcard $1/$(SRC_DIR)/*.c)

# $(call get-archive-file,dir) -> archive-list
get-archive-file = $(addsuffix .a,$(file < $1/$(LIBFILE)))

# $(call get-include-path,from-source) -> include-path
get-include-path = $(patsubst %$(SRC_DIR)/,%$(INC_DIR)/,$(dir $1))

# $(call create-symlink,base-dir,target-dir,name)
create-symlink = $(shell													\
	$(MKDIR) $1;															\
	$(TEST) -L $(strip $1)/$(strip $3) || 									\
	$(LN) $$(realpath -m --relative-to $1 $2) $(strip $1)/$(strip $3)		\
)

# $(call create-include-dir,base-dir)
create-include-dir = $(foreach d,$(file < $1/$(LIBFILE)),					\
	$(call create-symlink,													\
		$(patsubst %/,%,$1/$(INC_DIR)/$(dir $d)),							\
		$(LIB_DIR)/$d/$(INC_DIR),											\
		$(notdir $d)														\
	)																		\
)

# $(call get-number-of-libraries)
get-number-of-libraries = $(words 											\
	$(foreach u,$(wildcard $(LIB_DIR)/*),$(wildcard $u/*))					\
)

# $(call make-shared-library,name,soname,dir)
define make-shared-library
$(eval SRC := $(patsubst %/main.c,,$(call get-source-file,$3)))
$(eval OBJ := $(addprefix $(OUT_DIR)/,$(patsubst %.c,%.o,$(SRC))))
$(eval ARV := $(addprefix $(OUT_DIR)/$(LIB_DIR)/,$(call get-archive-file,$3)))

SOURCES += $(SRC)
OBJECTS += $(OBJ)
INCLUDES += $(wildcard $3/$(INC_DIR)/*.h)

CFLAGS += -fPIC

$(OUT_DIR)/$1:: $(BLDFILE)
	$(TOUCH) -c $(SRC) TEMP -d "$(SYNC_TIME)"

$(OUT_DIR)/$1:: $(OBJ) $(ARV)
	$(CC) -o $$@ $$^ $$(LDFLAGS) -shared $(LDLIBS) -Wl,-soname,$2

endef

# $(call make-library,name,dir)
define make-library
$(eval SRC := $(patsubst %/main.c,,$(call get-source-file,$2)))
$(eval OBJ := $(addprefix $(OUT_DIR)/,$(patsubst %.c,%.o,$(SRC))))
$(eval ARV := $(addprefix $(OUT_DIR)/$(LIB_DIR)/,$(call get-archive-file,$2)))

ifneq "$(SRC)" ""
SOURCES += $(SRC)
OBJECTS += $(OBJ)
INCLUDES += $(wildcard $2/$(INC_DIR)/*.h)

$(OUT_DIR)/$(LIB_DIR)/$1:: $(BLDFILE)
	$(TOUCH) -c $(SRC) TEMP -d "$(SYNC_TIME)"

$(OUT_DIR)/$(LIB_DIR)/$1:: $(OBJ) $(ARV)
	$(AR) $(ARFLAGS) $$@ $$^

else
$(OUT_DIR)/$(LIB_DIR)/$1:
	$(TOUCH) $$@

endif

endef

# $(call make-program,name,dir)
define make-program
$(eval SRC := $(call get-source-file,$2))
$(eval OBJ := $(addprefix $(OUT_DIR)/,$(patsubst %.c,%.o,$(SRC))))
$(eval ARV := $(addprefix $(OUT_DIR)/$(LIB_DIR)/,$(call get-archive-file,$2)))

SOURCES += $(SRC)
OBJECTS += $(OBJ)
INCLUDES += $(wildcard $2/$(INC_DIR)/*.h)

$(OUT_DIR)/$1:: $(BLDFILE)
	$(TOUCH) -c $(SRC) TEMP -d "$(SYNC_TIME)"

$(OUT_DIR)/$1:: $(OBJ) $(ARV)
	$(CC) -o $$@ $$^ $(LDLIBS)

endef

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
$(call create-include-dir,.)

$(foreach l,$(LIBRARIES),													\
	$(eval LIBRARIES := $(sort												\
		$(LIBRARIES) $(file < $(LIB_DIR)/$l/$(LIBFILE))						\
	))																		\
)

ifneq "$(words $(LIBRARIES))" "$(call get-number-of-libraries)"

download_libraries := $(foreach l,$(LIBRARIES),								\
	$(shell test -d $(LIB_DIR)/$l 											\
		 || git clone https://github.com/$l $(LIB_DIR)/$l)					\
	$(call create-include-dir,$(LIB_DIR)/$l)								\
)

.PHONY: FORCE
FORCE:

%:: FORCE
	@$(MAKE) $@

else # ifneq "$(words $(LIBRARIES))" "$(call get-number-of-libraries)"

create_output_dir := $(shell												\
	$(MKDIR) $(OUT_DIR);													\
	$(MKDIR) $(OUT_DIR)/$(SRC_DIR);											\
	for f in $(sort $(dir $(OBJECTS)));										\
	do																		\
		$(TEST) -d $$f 														\
			|| $(MKDIR) $$f;												\
	done;																	\
	for l in $(LIBRARIES);													\
	do																		\
		$(MKDIR) $(OUT_DIR)/$(LIB_DIR)/$$l/$(SRC_DIR);						\
	done																	\
)

# -----------------------------------------------------------------------------
# Rules 
# -----------------------------------------------------------------------------
# Libraries
$(foreach l,$(LIBRARIES),													\
	$(eval $(call make-library,$l.a,$(LIB_DIR)/$l))							\
)

# Output
ifneq ($(patsubst %.so,%,$(OUTPUT)),$(OUTPUT))
$(eval $(call make-shared-library,$(OUTPUT),$(SONAME),.))
else
$(eval $(call make-program,$(OUTPUT),.))
endif

DEPENDENCIES := $(patsubst %.o,%.d,$(OBJECTS))

# -----------------------------------------------------------------------------
# Recipes 
# -----------------------------------------------------------------------------
$(OBJECTS): $(OUT_DIR)/%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@ 								\
		  -I$(call get-include-path,$<)

$(DEPENDENCIES): $(OUT_DIR)/%.d: %.c
	@$(CC) $(CFLAGS) -I$(call get-include-path,$<)							\
		   $(CPPFLAGS) $(TARGET_ARCH) -MG -MM $<	| 						\
	$(SED) 's,\($(notdir $*)\.o\) *:,$(dir $@)\1 $@: ,' > $@.tmp
	@$(MV) $@.tmp $@

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
.PHONY: build
build: $(DEPENDENCIES) $(OUT_DIR)/$(OUTPUT)

.PHONY: compile
compile: $(OBJECTS)

.PHONY: update
update:
	for l in $(sort $(LIBRARIES));			\
	do										\
		(cd $(LIB_DIR)/$$l; git pull)		\
    done

.PHONY: help
help:
	@$(CAT) $(MAKEFILE_LIST)											|	\
	$(GREP) -v -e '^$$1' -v -e '^FORCE'									| 	\
	$(AWK) '/^[^.%][-A-Za-z0-9_]*:/											\
		   { print substr($$1, 1, length($$1) - 1) }'					|	\
	$(SORT)																|	\
	$(PR) --omit-pagination --width=80 --columns=4

.PHONY: tags
tags: $(SOURCES) $(INCLUDES)
	$(CTAGS) -f $(CTAGS_OUT) $^ 

.PHONY: cscope
cscope: $(SOURCES) $(INCLUDES)
	echo "$(SOURCES) $(INCLUDES)" > $(CSCOPE_FILE_OUT)
	$(CSCOPE) -i $(CSCOPE_FILE_OUT) -f $(CSCOPE_DB_OUT)

.PHONY: bear
bear: clean
	$(BEAR) --output $(COMPILE_DB_OUT) -- $(MAKE) all

.PHONY: all
all: build

.PHONY: clean
clean:
	$(RM) -r $(OUT_DIR)
	$(RM) -r $(addprefix $(INC_DIR)/,$(dir $(LIBRARIES)))

.PHONY: cleanall
cleanall: clean
	$(RM) -r $(LIB_DIR)
	$(RM) $(CSCOPE_DB_OUT) $(CSCOPE_FILE_OUT) $(CTAGS_OUT) $(COMPILE_DB_OUT)

.PHONY: variables
variables:
	# Variables: $(strip $(foreach v,$(.VARIABLES),							\
			$(if $(filter file,$(origin $v)),$v))							\
	)
	$(foreach g,$(MAKECMDGOALS),$(if $(filter-out variables,$g),$g: $($g)))

.PHONY: install
install:

.PHONY: run
run: $(OUT_DIR)/$(OUTPUT)
	@./$(OUT_DIR)/$(OUTPUT) $(file < $(RUNFILE))

.PHONY: example
example:
	$(MKDIR) $(SRC_DIR) $(INC_DIR) $(LIB_DIR) $(CNF_DIR)
	$(TOUCH) $(LIBFILE) $(RUNFILE)

	$(WGET) https://raw.githubusercontent.com/Cruzer-S/generic-makefile/main/$(SRC_DIR)/main.c
	$(WGET) https://raw.githubusercontent.com/Cruzer-S/generic-makefile/main/$(BLDFILE)

	$(MV) $(notdir $(BLDFILE)) $(CNF_DIR)/
	$(MV) main.c $(SRC_DIR)

# -----------------------------------------------------------------------------
# Include
# -----------------------------------------------------------------------------
ifeq "$(findstring $(MAKECMDGOALS),clean cleanall)" ""
-include $(DEPENDENCIES)
endif

endif # else of "$(words $(LIBRARIES))" "$(call get-number-of-libraries)"
