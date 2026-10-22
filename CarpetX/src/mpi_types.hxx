#ifndef CARPETX_CARPETX_MPI_TYPES_HXX
#define CARPETX_CARPETX_MPI_TYPES_HXX

#ifndef CARPETX_INTERNAL
#error                                                                         \
    "CarpetX internal header; include <carpetx.hxx> instead (see agent_docs/carpetx-public-api.md)"
#endif

#include <mpi.h>

namespace CarpetX {

template <typename T> struct mpi_datatype;

template <> struct mpi_datatype<char> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<short> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<int> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<long> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<long long> {
  static const MPI_Datatype value;
};

template <> struct mpi_datatype<unsigned char> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<unsigned short> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<unsigned int> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<unsigned long> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<unsigned long long> {
  static const MPI_Datatype value;
};

template <> struct mpi_datatype<float> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<double> {
  static const MPI_Datatype value;
};
template <> struct mpi_datatype<long double> {
  static const MPI_Datatype value;
};

template <typename T>
inline MPI_Datatype mpi_datatype_v = mpi_datatype<T>::value;

} // namespace CarpetX

#endif // #ifndef CARPETX_CARPETX_MPI_TYPES_HXX
