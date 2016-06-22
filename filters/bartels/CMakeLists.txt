PDAL_ADD_DRIVER(filter bartels "BartelsFilter.cpp" "BartelsFilter.hpp" objects)
set(PDAL_TARGET_OBJECTS ${PDAL_TARGET_OBJECTS} ${objects} PARENT_SCOPE)
