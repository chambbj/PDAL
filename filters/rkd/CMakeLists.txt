PDAL_ADD_DRIVER(filter rkd "RKDFilter.cpp" "RKDFilter.hpp" objects)
set(PDAL_TARGET_OBJECTS ${PDAL_TARGET_OBJECTS} ${objects} PARENT_SCOPE)
