#include "mig_axi_model.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace supra2::latency;

int main(int argc, char** argv) {
  try {
    std::string input_path;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "Usage: mig_axi_model --requests FILE.csv\n"
                  << "CSV columns: direction(R/W), byte_address, full_512b_beats\n";
        return 0;
      }
      if (arg == "--requests" && i + 1 < argc) input_path = argv[++i];
      else throw std::invalid_argument("expected --requests FILE.csv or --help");
    }
    if (input_path.empty()) throw std::invalid_argument("--requests is required");

    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("cannot open request file: " + input_path);
    MigAxiModel model;
    std::string line;
    std::uint64_t id = 1;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream row(line);
      std::string direction, address, beats;
      if (!std::getline(row, direction, ',') || !std::getline(row, address, ',') ||
          !std::getline(row, beats, ','))
        throw std::invalid_argument("malformed request row: " + line);
      AxiRequest request;
      request.id = id++;
      if (direction == "R" || direction == "READ") request.direction = AxiDirection::Read;
      else if (direction == "W" || direction == "WRITE") request.direction = AxiDirection::Write;
      else throw std::invalid_argument("direction must be R or W: " + direction);
      request.address = std::stoull(address, nullptr, 0);
      request.beats = static_cast<std::uint32_t>(std::stoul(beats, nullptr, 0));
      while (!model.submit(request)) model.step_ui();
    }
    model.run_until_idle();
    std::cout << model.summary() << '\n';
    for (const auto& completion : model.completions()) {
      std::cout << "request=" << completion.id
                << " direction=" << (completion.direction == AxiDirection::Read ? "R" : "W")
                << " submitted_ui=" << completion.submitted_ui
                << " completed_ui=" << completion.completed_ui
                << " latency_ui=" << completion.completed_ui - completion.submitted_ui
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
