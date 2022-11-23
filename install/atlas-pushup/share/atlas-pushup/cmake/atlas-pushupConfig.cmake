# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_atlas-pushup_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED atlas-pushup_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(atlas-pushup_FOUND FALSE)
  elseif(NOT atlas-pushup_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(atlas-pushup_FOUND FALSE)
  endif()
  return()
endif()
set(_atlas-pushup_CONFIG_INCLUDED TRUE)

# output package information
if(NOT atlas-pushup_FIND_QUIETLY)
  message(STATUS "Found atlas-pushup: 0.0.0 (${atlas-pushup_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'atlas-pushup' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${atlas-pushup_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(atlas-pushup_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${atlas-pushup_DIR}/${_extra}")
endforeach()
