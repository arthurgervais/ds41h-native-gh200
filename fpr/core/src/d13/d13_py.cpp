// d13_py.cpp -- stage 2a: the C++ D13 host (d13_host.h) and HSD prepare (hsd_prep.h) inside today's Python engine
// (C1 tail session, 14 Nov 2026). Module fpr_d13_cxx, built and installed by d13_cxx.py under PHYS_CXX_D13=1.
// Converts the engine's Python objects (serve StepBatch / SeqEntry, rowexp_ref CompactTables, HSD's sampling states) to
// the core's structs per call and returns Python values the existing code consumes. Decisions are C++; device work and
// the objects' identities stay Python.
#include <torch/extension.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "d13_host.h"
#include "hsd_prep.h"

namespace py = pybind11;
using namespace fpr;
using namespace fpr::d13;

namespace {

void ivec(py::handle o, std::vector<int64_t>* v) {   // None / list / tuple / numpy -> int64
  v->clear();
  if (!o || o.is_none()) return;
  if (py::isinstance<py::array>(o)) {
    auto a = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(o);
    if (!a) throw std::runtime_error("fpr_d13_cxx: not an integer array");
    const int64_t* p = a.data();
    v->assign(p, p + a.size());
    return;
  }
  for (auto x : o) v->push_back(py::cast<int64_t>(x));
}
template <class T>
void ivec_t(py::handle o, std::vector<T>* v) {
  std::vector<int64_t> t;
  ivec(o, &t);
  v->assign(t.begin(), t.end());
}
int64_t iattr(py::handle o, const char* k, int64_t dflt) {
  if (!py::hasattr(o, k)) return dflt;
  py::object x = o.attr(k);
  if (x.is_none()) return dflt;
  return py::cast<int64_t>(x);
}
std::string rid_str(py::handle x) { return py::str(x); }

enum : int { WANT_BT = 1, WANT_WT = 2, WANT_ROWS = 4 };

SeqEntry seq_of(py::handle se, int want) {
  SeqEntry e;
  e.rid = rid_str(se.attr("rid"));
  e.kind = py::cast<std::string>(se.attr("kind"));
  e.row0 = (int32_t)iattr(se, "row0", 0);
  e.n_rows = (int32_t)iattr(se, "n_rows", 0);
  e.start = iattr(se, "start", 0);
  e.num_computed = iattr(se, "num_computed", 0);
  e.cached_upto = iattr(se, "cached_upto", 0);
  e.replay_start = iattr(se, "replay_start", 0);
  e.last_chunk = py::cast<bool>(se.attr("last_chunk"));
  e.state_slot = (int32_t)iattr(se, "state_slot", -1);
  e.tail_row0 = (int32_t)iattr(se, "tail_row0", 0);
  e.n_blocks = iattr(se, "n_blocks", 0);
  e.score = py::cast<bool>(se.attr("score"));
  e.forced = py::cast<bool>(se.attr("forced"));
  e.uid = (int32_t)iattr(se, "uid", -1);
  e.known = iattr(se, "known", -1);
  e.wver = iattr(se, "wver", -1);
  e.inc = iattr(se, "inc", 0);
  if (want & WANT_BT) {
    auto bt = std::make_shared<std::vector<int64_t>>();
    ivec(se.attr("block_table"), bt.get());
    e.block_table = bt;
  } else {   // len() only (n_blocks fallback): a vector of the right size is enough for se.n_blocks or len(bt)
    py::object b = se.attr("block_table");
    e.block_table = std::make_shared<std::vector<int64_t>>(b.is_none() ? 0 : py::len(b), 0);
  }
  if (want & WANT_WT) {
    auto wt = std::make_shared<WindowTable>();
    py::object w = se.attr("window_table");
    if (w.is_none()) {
    } else if (py::isinstance<py::dict>(w)) {   // the Python KV manager: {block: page}
      for (auto kv : py::reinterpret_borrow<py::dict>(w)) wt->items.push_back({py::cast<int64_t>(kv.first), py::cast<int64_t>(kv.second)});
    } else {   // c1-sync's C++ KV manager (PHYS_CXX_KV): fpr_kvtables.Window, items() = [(block, page)] in its table order
      for (auto it : w.attr("items")()) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(it);
        wt->items.push_back({py::cast<int64_t>(t[0]), py::cast<int64_t>(t[1])});
      }
    }
    e.window_table = wt;
  }
  return e;
}

StepBatch batch_of(py::handle sb, int want) {
  StepBatch b;
  b.rank = (int32_t)iattr(sb, "rank", 0);
  b.step = iattr(sb, "step", 0);
  b.n_rows = (int32_t)iattr(sb, "n_rows", 0);
  if (want & WANT_ROWS) {
    ivec(sb.attr("positions"), &b.positions);
    ivec_t(sb.attr("seq_slot"), &b.seq_slot);
    ivec(sb.attr("paged_slot"), &b.paged_slot);
    ivec(sb.attr("window_slot"), &b.window_slot);
    ivec_t(sb.attr("tail_rows"), &b.tail_rows);
    ivec_t(sb.attr("sample_rows"), &b.sample_rows);
  }
  py::object cp = sb.attr("copies");
  b.copies.resize(cp.is_none() ? 0 : py::len(cp));
  for (auto se : sb.attr("seqs")) b.seqs.push_back(seq_of(se, want));
  return b;
}

CompactTables ct_of(py::handle ct) {
  CompactTables c;
  c.rank = (int32_t)iattr(ct, "rank", 0);
  c.step = iattr(ct, "step", 0);
  c.k1 = (int32_t)iattr(ct, "k1", 0);
  ivec(ct.attr("slot"), &c.slot); ivec(ct.attr("row0"), &c.row0); ivec(ct.attr("n_rows"), &c.n_rows);
  ivec(ct.attr("kind"), &c.kind); ivec(ct.attr("start"), &c.start); ivec(ct.attr("replay_start"), &c.replay_start);
  ivec(ct.attr("cached_upto"), &c.cached_upto);
  ivec(ct.attr("win_slot"), &c.win_slot); ivec(ct.attr("win_blk"), &c.win_blk); ivec(ct.attr("win_page"), &c.win_page);
  ivec(ct.attr("bt_upd_slot"), &c.bt_upd_slot); ivec(ct.attr("bt_upd_start"), &c.bt_upd_start);
  ivec(ct.attr("bt_upd_len"), &c.bt_upd_len); ivec(ct.attr("bt_upd_pages"), &c.bt_upd_pages);
  ivec(ct.attr("tokens"), &c.tokens); ivec(ct.attr("sample_idx"), &c.sample_idx);
  ivec(ct.attr("uid"), &c.uid); ivec(ct.attr("last_chunk"), &c.last_chunk); ivec(ct.attr("score"), &c.score);
  ivec(ct.attr("forced"), &c.forced);
  int64_t n = 0;
  for (auto x : c.n_rows) n += x;
  c.n = n;
  return c;
}

template <class T>
py::array_t<T> arr(const std::vector<T>& v) {
  py::array_t<T> a((py::ssize_t)v.size());
  if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

[[noreturn]] void raise_py(const std::string& err) {   // the Python exception types _build raises
  if (err.find("exceeds the largest bucket") != std::string::npos) throw py::value_error(err);
  PyErr_SetString(PyExc_AssertionError, err.c_str());
  throw py::error_already_set();
}

class Host {
 public:
  Host(py::dict cfg, py::dict hcfg) : h_(cfg_of(cfg), hcfg_of(hcfg)) {}

  std::string async_ok(py::sequence batches, py::sequence cts, bool graphs) {
    std::vector<StepBatch> b;
    std::vector<CompactTables> c;
    for (auto x : batches) b.push_back(batch_of(x, 0));
    for (auto x : cts) c.push_back(ct_of(x));
    h_.set_graphs(graphs);
    return h_.async_ok(b, c);
  }

  py::tuple delta_rows(py::handle ct) {   // d13._delta_rows: (nu, rows, cols, pages) (nu == 0: None x 3)
    Deltas d;
    h_.deltas(0, ct_of(ct), &d);
    if (!d.nu) return py::make_tuple(0, py::none(), py::none(), py::none());
    return py::make_tuple(d.nu, arr(d.rows), arr(d.cols), arr(d.pages));
  }

  py::object owner_sig(int r, py::handle sb, py::handle ct, py::sequence prev) {
    std::vector<PrevOf> pv;
    for (auto p : prev) {
      py::tuple t = py::reinterpret_borrow<py::tuple>(p);
      pv.push_back(PrevOf{py::cast<std::string>(t[0]), py::cast<bool>(t[1]), py::cast<int64_t>(t[2]), py::cast<int64_t>(t[3])});
    }
    StepBatch b = batch_of(sb, 0);
    auto s = h_.owner_sig(r, b, ct_of(ct), pv);
    if (!s) return py::none();
    py::list seqs;   // (seq index, state slot, row0, n_rows, n_blocks, prev index): d13_cxx maps the index to se.rid
    for (size_t k = 0; k < s->seqs.size(); ++k) {
      const auto& q = s->seqs[k];
      seqs.append(py::make_tuple((int64_t)k, q.state_slot, q.row0, q.n_rows, q.n_blocks, q.prev_index));
    }
    return py::make_tuple(s->n, seqs, py::tuple(py::cast(s->sample_idx)));
  }

  bool empty_eligible(py::handle sb, py::handle ct) { return h_.empty_eligible(batch_of(sb, 0), ct_of(ct)); }

  py::list fb_rows(py::sequence batches, py::sequence starts) {   // d13._fb_rows's host half
    std::vector<StepBatch> b, rb;
    for (auto x : batches) b.push_back(batch_of(x, WANT_BT | WANT_WT | WANT_ROWS));
    std::vector<std::vector<int64_t>> st;
    for (auto x : starts) {
      st.emplace_back();
      ivec(x, &st.back());
    }
    std::string err;
    if (h_.fallback_resolve(b, st, &rb, &err) != Status::OK) raise_py(err);
    py::list out;
    for (size_t r = 0; r < rb.size(); ++r) {
      py::list upd;   // (k, start, num_computed, n_blocks) of the sequences replaced (device-resolved starts)
      for (size_t k = 0; k < rb[r].seqs.size(); ++k)
        if (b[r].seqs[k].start < 0)
          upd.append(py::make_tuple((int64_t)k, rb[r].seqs[k].start, rb[r].seqs[k].num_computed, rb[r].seqs[k].n_blocks));
      out.append(py::make_tuple(upd, (int64_t)rb[r].seqs.size(), py::cast(rb[r].positions), py::cast(rb[r].paged_slot),
                                py::cast(rb[r].window_slot)));
    }
    return out;
  }

  // ---- HSD prepare
  void hsd_bind(py::sequence blobs, int64_t bm_ptr) {
    HsdPrep& p = h_.hsd();
    for (int i = 0; i < 2; ++i) {
      py::sequence row = py::reinterpret_borrow<py::sequence>(blobs[i]);
      for (int r = 0; r < W; ++r) p.bind(i, r, (uint8_t*)py::cast<int64_t>(row[r]));
    }
    if (bm_ptr) p.bind_bm((int32_t*)bm_ptr);
  }
  void hsd_restore(py::array_t<float, py::array::c_style | py::array::forcecast> temp,
                   py::array_t<int64_t, py::array::c_style | py::array::forcecast> seeds, py::sequence keys) {
    std::vector<float> t(temp.data(), temp.data() + temp.size());
    std::vector<int64_t> s(seeds.data(), seeds.data() + seeds.size());
    std::vector<uint64_t> k;
    for (auto x : keys) k.push_back(py::cast<uint64_t>(x));
    h_.hsd().restore(t, s, k);
  }
  py::list hsd_layout() {
    py::list o;
    for (const auto& f : h_.hsd().layout()) o.append(py::make_tuple(f.name, f.off, f.count));
    return o;
  }
  int64_t hsd_bytes() { return h_.hsd_blob_bytes(); }
  int64_t hsd_shared_end() { return h_.hsd().shared_end(); }

  // states[r][k] = (temperature, seed, drafts_valid, num_acceptable or None, id(state)) or None (not a drafting row)
  py::object hsd_build(int hi, py::sequence batches, py::sequence states, py::object bitmasks) {
    std::vector<StepBatch> b;
    for (auto x : batches) b.push_back(batch_of(x, WANT_WT | WANT_ROWS));
    std::vector<std::vector<HsdState>> sv(b.size());
    for (size_t r = 0; r < b.size(); ++r) {
      py::sequence sr = states[r];
      sv[r].resize(b[r].seqs.size());
      for (size_t k = 0; k < sv[r].size() && k < (size_t)py::len(sr); ++k) {
        py::object t = sr[k];
        if (t.is_none()) continue;
        py::tuple q = py::reinterpret_borrow<py::tuple>(t);
        HsdState& s = sv[r][k];
        s.temperature = py::cast<double>(q[0]);
        s.seed = py::cast<int64_t>(q[1]);
        s.drafts_valid = py::cast<bool>(q[2]);
        s.has_num_acceptable = !q[3].is_none();
        if (s.has_num_acceptable) s.num_acceptable = py::cast<int64_t>(q[3]);
        s.ident = py::cast<uint64_t>(q[4]);
      }
    }
    std::vector<HsdBitmask> bms;
    std::vector<py::array_t<int32_t, py::array::c_style | py::array::forcecast>> keep;
    const bool has_bm = !bitmasks.is_none();
    if (has_bm)
      for (auto x : bitmasks) {
        HsdBitmask m;
        if (!x.is_none()) {
          py::object a = py::hasattr(x, "numpy") ? x.attr("numpy")() : py::reinterpret_borrow<py::object>(x);
          keep.push_back(py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(a));
          m.n_s = keep.back().shape(0);
          m.words = keep.back().shape(1);
          m.rows = keep.back().data();
        }
        bms.push_back(m);
      }
    HsdStep cur;
    std::string err;
    h_.hsd().set_hi(hi);
    if (h_.hsd_prepare(b, sv, has_bm ? &bms : nullptr, &cur, &err) != Status::OK) raise_py(err);
    if (cur.v0) return py::none();
    py::list dseqs;
    for (const auto& x : cur.dseqs) dseqs.append(py::cast(x));
    py::dict d;
    d["hi"] = cur.hi; d["S_pad"] = cur.S_pad; d["D_pad"] = cur.D_pad; d["NT"] = cur.NT; d["n_req"] = cur.n_req;
    d["T"] = cur.T; d["n_s"] = cur.n_s; d["n_seq"] = cur.n_seq; d["n_t"] = cur.n_t; d["step"] = cur.step;
    d["dseqs"] = dseqs; d["has_bm"] = cur.has_bm;
    return d;
  }

 private:
  static Config cfg_of(py::dict c) {
    Config x;
    x.graphs = py::cast<bool>(c["graphs"]);
    x.largest_bucket = py::cast<int64_t>(c["largest_bucket"]);
    x.graph_max_ctx = py::cast<int64_t>(c["graph_max_ctx"]);
    x.k1 = py::cast<int64_t>(c["k1"]);
    x.delta_cap = py::cast<int64_t>(c["delta_cap"]);
    x.fb_reuse = py::cast<bool>(c["fb_reuse"]);
    py::sequence sc = c["scratch"];
    for (int r = 0; r < W && r < (int)py::len(sc); ++r) x.scratch[r] = py::cast<int64_t>(sc[r]);
    return x;
  }
  static HsdConfig hcfg_of(py::dict c) {
    HsdConfig h;
    if (c.empty()) {
      h.seq_buckets = {8};
      h.row_buckets = {32};
      return h;
    }
    h.k = py::cast<int64_t>(c["k"]);
    h.max_seqs = py::cast<int64_t>(c["max_seqs"]);
    h.seq_buckets = py::cast<std::vector<int64_t>>(c["seq_buckets"]);
    h.row_buckets = py::cast<std::vector<int64_t>>(c["row_buckets"]);
    h.nt_max = py::cast<int64_t>(c["nt_max"]);
    h.pool = py::cast<bool>(c["pool"]);
    h.window = py::cast<int64_t>(c["window"]);
    h.ring = py::cast<int64_t>(c["ring"]);
    return h;
  }
  D13Host h_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Host>(m, "Host")
      .def(py::init<py::dict, py::dict>())
      .def("async_ok", &Host::async_ok)
      .def("delta_rows", &Host::delta_rows)
      .def("owner_sig", &Host::owner_sig)
      .def("empty_eligible", &Host::empty_eligible)
      .def("fb_rows", &Host::fb_rows)
      .def("hsd_bind", &Host::hsd_bind)
      .def("hsd_restore", &Host::hsd_restore)
      .def("hsd_layout", &Host::hsd_layout)
      .def("hsd_bytes", &Host::hsd_bytes)
      .def("hsd_shared_end", &Host::hsd_shared_end)
      .def("hsd_build", &Host::hsd_build);
}
