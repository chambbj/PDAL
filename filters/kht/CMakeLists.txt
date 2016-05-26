PDAL_ADD_DRIVER(filter kht "KHTFilter.cpp" "KHTFilter.hpp" objects)
set(PDAL_TARGET_OBJECTS ${PDAL_TARGET_OBJECTS} ${objects} PARENT_SCOPE)
