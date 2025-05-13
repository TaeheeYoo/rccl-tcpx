#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.71])
AC_INIT([FULL-PACKAGE-NAME], [VERSION], [BUG-REPORT-ADDRESS])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.
# FIXME: Replace 'main' with a function in '-lamdhip64':
AC_CHECK_LIB([amdhip64], [main])
# FIXME: Replace 'main' with a function in '-lynl':
AC_CHECK_LIB([ynl], [main])

AC_PROG_RANLIB
AM_INIT_AUTOMAKE( [foreign -Wall] )
AC_CONFIG_FILES( [Makefile src/Makefile] )

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AC_OUTPUT
