#!/bin/bash
# copy_root.sh SRC DST (c1-overlap, 17 Nov 2026; main 01:20: THE way to copy a gate root, after two prebuilt.env contaminations
# on 16-17 Nov: c1-tail relB1pc -> relB1, c128attn relB1fmu -> relB1f). Login node only.
#   1. cp -a SRC DST (DST must not exist; neither may be the served root)
#   2. every TEXT file under DST naming SRC/ (prebuilt.env, build.ninja, *.env, *.sh, *.py, *.json ...) -> DST/
#   3. drop every lock, .ninja_deps and .ninja_log (binary dep logs still name SRC)
#   4. ninja in EVERY build dir (dirs with a build.ninja), sourced from DST: activate.sh, env.sh, DST's prebuilt.env
#      -> every compiled object is rebuilt from DST's own sources, so nothing JITs later on a GPU node
#   5. verify, REFUSE (exit 1, DST left for inspection, file DST/COPY_ROOT_REFUSED) unless:
#        text files naming SRC = 0; build.ninja not naming DST = 0; locks = 0; every ninja rc 0;
#        prebuilt.env names DST and not SRC
#      Binary files still naming SRC (e.g. a .so outside any build dir with __FILE__ strings) are LISTED, not refused,
#      unless they sit inside a build dir (then the rebuild failed to replace them -> refuse).
#   On success: DST/COPY_ROOT_OK with SRC, date and counts. Gate scripts can guard on it.
#   (v2) step 4 first drops every *.so / *.o in a build dir that its build.ninja does not produce (torch _vN leftovers).
# Options: MAX_JOBS (default 8). Refuses off a clariden-ln* login node or inside a Slurm job (FORCE_GPU_HOST=1 overrides).
set -u
SRC=$(readlink -f "${1:?usage: copy_root.sh SRC DST}"); DST_IN="${2:?usage: copy_root.sh SRC DST}"
H=${DS41H_HOME:?set DS41H_HOME}
die() { echo "copy_root: REFUSED: $*"; [ -d "${DST:-/nonexistent}" ] && echo "$* ($(date +%F_%T))" > "$DST/COPY_ROOT_REFUSED"; exit 1; }
[ -d "$SRC" ] || die "SRC $SRC is not a directory"
[ -e "$DST_IN" ] && { echo "copy_root: REFUSED: DST $DST_IN exists (never overwrite a root)"; exit 1; }
mkdir -p "$(dirname "$DST_IN")"; DST=$(readlink -f "$(dirname "$DST_IN")")/$(basename "$DST_IN")
case "$DST/" in "$SRC/"*) echo "copy_root: REFUSED: DST inside SRC"; exit 1;; esac
grep -q "^FPR_RELEASE=$DST\$" $H/slot.env 2>/dev/null && { echo "copy_root: REFUSED: DST is the served root"; exit 1; }
case "$(hostname)" in clariden-ln*) ;; *) [ "${FORCE_GPU_HOST:-0}" = 1 ] || { echo "copy_root: REFUSED: not a Clariden login node ($(hostname)); never build on a compute node"; exit 1; };; esac
[ -n "${SLURM_JOB_ID:-}" ] && [ "${FORCE_GPU_HOST:-0}" != 1 ] && { echo "copy_root: REFUSED: inside a Slurm job"; exit 1; }
echo "copy_root: $SRC -> $DST ($(date +%T))"
SRE="${SRC//./\\.}([/\"' :;=]|\$)"   # SRC as a whole path component (relB1 must not match relB1f)
cp -a "$SRC" "$DST" || die "cp -a failed"
chmod -R u+w "$DST" || die "chmod u+w DST failed"   # main 18 Nov: frozen (read-only) SRC trees keep -r modes in DST; ninja logs must be writable
rm -f "$DST/COPY_ROOT_OK" "$DST/COPY_ROOT_REFUSED"
# 2. text files naming SRC/ (grep -I skips binaries)
nt=0
while IFS= read -r f; do sed -i "s#$SRC/#$DST/#g; s#$SRC\$#$DST#g; s#$SRC\([\"' :;]\)#$DST\1#g" "$f"; nt=$((nt + 1)); done < <(grep -rlIE -- "$SRE" "$DST" 2>/dev/null)
echo "copy_root: re-pointed $nt text files"
# 3. locks + ninja dep logs
find "$DST" \( -name lock -o -name .ninja_deps -o -name .ninja_log \) -type f -delete
# 4. rebuild every build dir from DST's sources
(
  cd "$DST" || exit 9
  [ -f clariden/activate.sh ] && source clariden/activate.sh >/dev/null 2>&1
  [ -f scripts/env.sh ] && source scripts/env.sh >/dev/null 2>&1
  [ -f fpr/build/prebuilt.env ] && source fpr/build/prebuilt.env
  export CUDA_VISIBLE_DEVICES= MAX_JOBS=${MAX_JOBS:-8}
  ok=0; bad=0
  for b in $(find "$DST" -name build.ninja -not -path '*.stale-*'); do
    d=$(dirname "$b")
    # (v2, c128moe 17 Nov: torch leaves name_v1.so next to name.so; ninja targets one, the other survives naming SRC) drop every
    # *.so / *.o in the build dir that is NOT an output of its build.ninja, before ninja (listed for the log)
    tg=$(cd "$d" && ninja -t targets all 2>/dev/null | sed 's/: .*//')
    for x in "$d"/*.so "$d"/*.o; do
      [ -e "$x" ] || continue
      echo "$tg" | grep -qxF -- "$(basename "$x")" || echo "$tg" | grep -qxF -- "$x" || { echo "copy_root: removed non-target ${x#$DST/}"; rm -f "$x"; }
    done
    if (cd "$d" && timeout 1800 ninja -j ${MAX_JOBS:-8} > "$d/.copy_root_ninja.log" 2>&1); then ok=$((ok + 1))
    else bad=$((bad + 1)); echo "copy_root: NINJA FAILED ${d#$DST/}: $(tail -2 "$d/.copy_root_ninja.log" | tr '\n' ' ' | cut -c1-200)"; fi
  done
  echo "copy_root: ninja ok $ok failed $bad"; echo "$bad" > "$DST/.copy_root_bad"
)
bad=$(cat "$DST/.copy_root_bad" 2>/dev/null || echo X); rm -f "$DST/.copy_root_bad"
# 5. verify
text_refs=$(grep -rlIE -- "$SRE" "$DST" 2>/dev/null | wc -l)
stale=$(find "$DST" -name build.ninja -not -path '*.stale-*' | xargs -r grep -L "$DST/" 2>/dev/null | wc -l)
locks=$(find "$DST" -name lock | wc -l)
bin_refs=$(grep -rlE -- "$SRE" "$DST" 2>/dev/null | grep -v "/\.copy_root_ninja\.log$")
bdirs=$(find "$DST" -name build.ninja -not -path '*.stale-*' -printf '%h\n')
bin_in_build=0
for f in $bin_refs; do for d in $bdirs; do case "$f" in "$d"/*) bin_in_build=$((bin_in_build + 1)); echo "copy_root: binary in a build dir still names SRC: ${f#$DST/}";; esac; done; done
pe=$DST/fpr/build/prebuilt.env; pe_ok=1
[ -f "$pe" ] && { grep -qE -- "$SRE" "$pe" && pe_ok=0; grep -q -- "$DST/" "$pe" || pe_ok=0; }
echo "copy_root: text refs $text_refs, stale build.ninja $stale, locks $locks, ninja failed $bad, binary refs in build dirs $bin_in_build, prebuilt.env ok $pe_ok"
[ -n "$bin_refs" ] && echo "$bin_refs" | sed "s#^$DST/#copy_root: note: binary outside build dirs names SRC: #" | head -20
[ "$text_refs" = 0 ] && [ "$stale" = 0 ] && [ "$locks" = 0 ] && [ "$bad" = 0 ] && [ "$bin_in_build" = 0 ] && [ "$pe_ok" = 1 ] \
  || die "verification failed (text refs $text_refs, stale $stale, locks $locks, ninja failed $bad, build-dir binaries $bin_in_build, prebuilt.env ok $pe_ok)"
echo "SRC=$SRC DST=$DST date=$(date +%F_%T) text_repointed=$nt" > "$DST/COPY_ROOT_OK"
echo "copy_root: OK $DST ($(date +%T))"
