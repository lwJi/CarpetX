#include "util/mpi_types.hxx"

#include <mpi.h>

namespace CarpetX {

const MPI_Datatype mpi_datatype<char>::value = MPI_CHAR;
const MPI_Datatype mpi_datatype<short>::value = MPI_SHORT;
const MPI_Datatype mpi_datatype<int>::value = MPI_INT;
const MPI_Datatype mpi_datatype<long>::value = MPI_LONG;
const MPI_Datatype mpi_datatype<long long>::value = MPI_LONG_LONG;

const MPI_Datatype mpi_datatype<unsigned char>::value = MPI_UNSIGNED_CHAR;
const MPI_Datatype mpi_datatype<unsigned short>::value = MPI_UNSIGNED_SHORT;
const MPI_Datatype mpi_datatype<unsigned int>::value = MPI_UNSIGNED;
const MPI_Datatype mpi_datatype<unsigned long>::value = MPI_UNSIGNED_LONG;
const MPI_Datatype mpi_datatype<unsigned long long>::value =
    MPI_UNSIGNED_LONG_LONG;

const MPI_Datatype mpi_datatype<float>::value = MPI_FLOAT;
const MPI_Datatype mpi_datatype<double>::value = MPI_DOUBLE;
const MPI_Datatype mpi_datatype<long double>::value = MPI_LONG_DOUBLE;

} // namespace CarpetX
