PDAL_ADD_DRIVER(filter resample ResampleFilter.cpp ResampleFilter.hpp objects)
set(PDAL_TARGET_OBJECTS ${PDAL_TARGET_OBJECTS} ${objects} PARENT_SCOPE)
