#ifndef SPUTTERER_TOML_H
#define SPUTTERER_TOML_H

#define TOML_HEADER_ONLY 0
// defines to get TOML working with nvcc
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN 1
#define TOML_RETURN_BOOL_FROM_FOR_EACH_BROKEN_ACKNOWLEDGED 1

// no serialization needed
#define TOML_ENABLE_FORMATTERS 0

#include <toml++/toml.hpp>

#endif

