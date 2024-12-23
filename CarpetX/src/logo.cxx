#include "logo.hxx"

#include <sstream>
#include <string>
#include <vector>

namespace CarpetX {

std::string logo() {

  std::ostringstream buf;

  // buf << "                                                               \n";
  // buf << "    ██████╗ █████╗ ██████╗ ██████╗ ███████╗████████╗██╗  ██╗   \n";
  // buf << "   ██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝╚██╗██╔╝   \n";
  // buf << "   ██║     ███████║██████╔╝██████╔╝█████╗     ██║    ╚███╔╝    \n";
  // buf << "   ██║     ██╔══██║██╔══██╗██╔═══╝ ██╔══╝     ██║    ██╔██╗    \n";
  // buf << "   ╚██████╗██║  ██║██║  ██║██║     ███████╗   ██║   ██╔╝ ██╗   \n";
  // buf << "    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝   \n";
  // buf << "                                                               \n";

  buf << "                                                            \n";
  buf << "   _______ _______  ______  _____  _______ _______ _     _  \n";
  buf << "   |       |_____| |_____/ |_____] |______    |     \\___/   \n";
  buf << "   |_____  |     | |    \\_ |       |______    |    _/   \\_  \n";
  buf << "                                                            \n";

  return buf.str();
}
} // namespace CarpetX
