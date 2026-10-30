#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Cycle = std::uint64_t;
struct Shape { const char* name; int m, n, k; Cycle reference; };
// 固定为现有非 PROGRAMMED_TILES、M=256 单 block RTL 的七个 QLinear 阶段。
constexpr Shape shapes[] = {{"QKV",256,1728,576,531053}, {"SPROJ",256,576,576,290212},
  {"CQ",256,576,576,288335}, {"CKV",32,1152,576,96691},
  {"CPROJ",256,576,576,290044}, {"MLP0",256,2304,576,652317},
  {"MLP2",256,576,2304,808012}};

struct Options { int frontend_row_576; int epi_row_ii; Cycle wait_acc; bool frontend_overlap, drain_overlap; };
struct Result { Cycle total, frontend_end, last_array, last_drain; int tiles; };

// 对每个 M tile 准备一次，所有 N tile 共用输入，直到最后一次阵列读取完成。
// 前端双缓冲时，M tile i 的缓冲只在 i-2 的最后一次阵列使用完成后复用。
// 读出并行时采用两个结果快照槽：阵列启动前确保对应槽已被旧读出释放。
// 这是一种保守的槽分配策略；每次完整捕获假设已包含于 wait_acc，不额外计时。
Result run(const Shape& s, Options o) {
  if (s.m <= 0 || s.n <= 0 || s.k <= 0 || s.m%32 || s.n%32 ||
      o.frontend_row_576 <= 0 || o.epi_row_ii <= 0 || o.wait_acc == 0)
    throw std::invalid_argument("invalid shape or options");
  const int mt = s.m/32, nt = s.n/32;
  // 前端周期是假设的敏感性输入，并非 RTL 计数；这里只示意按 K 比例变化。
  const Cycle frontend = Cycle(32)*o.frontend_row_576*s.k/576;
  const Cycle array = 1 + s.k + 32 + 32 + 3 + o.wait_acc;
  const Cycle drain = Cycle(32)*o.epi_row_ii;
  std::vector<Cycle> ready(mt), release(mt), snap_release(mt*nt);
  Cycle front_available = 0, array_available = 0, drain_available = 0;
  if (!o.frontend_overlap) {
    for (int m=0; m<mt; ++m) { front_available += frontend; ready[m]=front_available; }
    array_available = front_available;
  }
  for (int m=0; m<mt; ++m) {
    if (o.frontend_overlap) {
      Cycle begin = front_available;
      if (m >= 2) begin = std::max(begin, release[m-2]);
      ready[m] = begin + frontend;
      front_available = ready[m];
    }
    for (int n=0; n<nt; ++n) {
      const int index=m*nt+n;
      Cycle begin=std::max(array_available, ready[m]);
      if (o.drain_overlap && index >= 2) begin=std::max(begin, snap_release[index-2]);
      const Cycle end=begin+array;
      const Cycle drain_end=std::max(end, drain_available)+drain;
      snap_release[index]=drain_end;
      drain_available=drain_end;
      array_available=o.drain_overlap ? end : drain_end;
      release[m]=end;
    }
  }
  return {std::max(array_available,drain_available),front_available,
          release.back(),drain_available,mt*nt};
}

void test() {
  Shape s{"test",96,64,576,0};
  Options base{64,1,1,false,false};
  auto b=run(s,base);
  // 3 个 M tile、每个两个 N tile，基线前端全部完成后串行计算和输出。
  if (b.total != 3*2048+6*(645+32) || b.tiles != 6) throw std::runtime_error("baseline accounting");
  auto a=base; a.drain_overlap=true;
  auto d=run(s,a);
  if (d.total != 3*2048+6*645+32 || d.total>=b.total) throw std::runtime_error("drain overlap");
  a.frontend_overlap=true;
  if (run(s,a).total>d.total) throw std::runtime_error("frontend overlap");
  a.epi_row_ii=100;
  if (run(s,a).total<=run(s,base).total) throw std::runtime_error("slow drain backpressure");
  Shape one{"single",32,32,576,0};
  if (run(one,a).total != 2048+645+3200) throw std::runtime_error("single tile");
}

int main(int argc,char** argv) try {
  if (argc==2 && std::string(argv[1])=="--self-test") { test(); return 0; }
  if (argc!=2) { std::cerr << "usage: gemm_architecture_model <report.csv> | --self-test\n"; return 1; }
  std::ofstream f(argv[1]); if(!f) throw std::runtime_error("cannot open report");
  f << "stage,m,n,k,tiles,frontend_row_576,epi_row_ii,wait_acc,architecture,local_cycles,baseline_cycles,local_savings,reference_stage_cycles,baseline_exceeds_reference,activation_buffer_bytes,snapshot_bytes\n";
  for(const auto& s:shapes) for(int row:{16,64,256}) for(int ii:{1,2,4}) {
    Options base{row,ii,1,false,false}; const auto baseline=run(s,base);
    for(int design=0;design<4;++design) {
      Options o=base; o.frontend_overlap=(design&1)!=0; o.drain_overlap=(design&2)!=0;
      auto result=run(s,o);
      const char* name=design==0?"serial":design==1?"frontend_pingpong":design==2?"drain_pingpong":"both_pingpong";
      f << s.name << ',' << s.m << ',' << s.n << ',' << s.k << ',' << result.tiles << ','
        << row << ',' << ii << ",1," << name << ',' << result.total << ',' << baseline.total << ','
        << baseline.total-result.total << ',' << s.reference << ',' << (baseline.total>s.reference?1:0) << ','
        << (o.frontend_overlap?2*32*s.k:s.m*s.k) << ',' << (o.drain_overlap?8192:4096) << '\n';
    }
  }
  std::cout << "Uncalibrated GEMM architecture exploration: " << argv[1] << '\n';
  return 0;
} catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
