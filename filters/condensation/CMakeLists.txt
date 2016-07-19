PDAL_ADD_DRIVER(filter condensation "CondensationFilter.cpp" "CondensationFilter.hpp" objects)
set(PDAL_TARGET_OBJECTS ${PDAL_TARGET_OBJECTS} ${objects} PARENT_SCOPE)
