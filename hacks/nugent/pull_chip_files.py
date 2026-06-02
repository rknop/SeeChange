#!/usr/bin/env python3
"""Pull missing image+sources+wcs trios from NERSC for an LS4 chip-list file.

Given a ``*.list`` file with one image-relative-path per line in the format::

    NIGHT/TIME/<basename>.image.fits.fz

this script:

  1.  Inventories which of the three companion files (``image.fits.fz``,
      ``sources_*.fits``, ``wcs_*.txt``) are present locally and at NERSC.
  2.  For entries where a complete trio is achievable, **pulls** any
      components that are missing locally.
  3.  For entries where the trio is *not* achievable (sources or wcs
      missing both locally and at NERSC), **deletes** any locally-present
      components so we don't keep stranded images around.

Tradeoffs:
  * One ``ssh ... find`` call to inventory the remote archive (avoids
    218*3 round-trips).
  * One ``ssh ... tar`` pipe to pull all missing files (one connection,
    flattening directory structure with ``--transform``).

Use ``--dry-run`` to preview without changing anything.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


KIND_IMAGE = "image"
KIND_SOURCES = "sources"
KIND_WCS = "wcs"
KINDS = (KIND_IMAGE, KIND_SOURCES, KIND_WCS)

LOCAL_GLOB_FOR_KIND = {
    KIND_IMAGE: "{base}.image.fits.fz",
    KIND_SOURCES: "{base}.sources_*.fits",
    KIND_WCS: "{base}.wcs_*.txt",
}


def parse_list(list_file):
    """Yield (night, time, basename) tuples from the *.list file."""
    for raw in Path(list_file).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("/")
        if len(parts) < 3:
            print(f"  skipping malformed line: {raw!r}", file=sys.stderr)
            continue
        night, time, fname = parts[-3], parts[-2], parts[-1]
        if not fname.endswith(".image.fits.fz"):
            print(f"  skipping non-image line: {raw!r}", file=sys.stderr)
            continue
        basename = fname[: -len(".image.fits.fz")]
        yield night, time, basename


def infer_chip(list_file):
    """Best-effort chip name inferred from the list filename, e.g. SE_C from SE_C.list."""
    name = Path(list_file).stem  # 'SE_C.list' -> 'SE_C'
    m = re.fullmatch(r"([NS][EW])_([A-H])", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def kind_of(filename):
    """Return KIND_IMAGE/SOURCES/WCS or None for an LS4-style filename."""
    if filename.endswith(".image.fits.fz"):
        return KIND_IMAGE
    if re.search(r"\.sources_[A-Za-z0-9]+\.fits$", filename):
        return KIND_SOURCES
    if re.search(r"\.wcs_[A-Za-z0-9]+\.txt$", filename):
        return KIND_WCS
    return None


def basename_of(filename):
    """Strip the kind-specific suffix to recover the LS4 basename."""
    if filename.endswith(".image.fits.fz"):
        return filename[: -len(".image.fits.fz")]
    m = re.match(r"^(.*?)\.(?:sources|wcs)_[A-Za-z0-9]+\.(?:fits|txt)$", filename)
    return m.group(1) if m else None


def remote_inventory(ssh_host, remote_base, chip):
    """Return dict[basename][kind] -> full_remote_path via a single ssh+find."""
    if chip:
        # Restrict the find to this chip to keep output small.
        name_clauses = [
            f"-name 'ls4_*_{chip}_*.image.fits.fz'",
            f"-name 'ls4_*_{chip}_*.sources_*.fits'",
            f"-name 'ls4_*_{chip}_*.wcs_*.txt'",
        ]
    else:
        name_clauses = [
            "-name 'ls4_*.image.fits.fz'",
            "-name 'ls4_*.sources_*.fits'",
            "-name 'ls4_*.wcs_*.txt'",
        ]
    or_clauses = " -o ".join(name_clauses)
    remote_cmd = (
        f"find {shlex.quote(remote_base)} -type f \\( {or_clauses} \\) "
        "-printf '%p\\n' 2>/dev/null"
    )
    print(f"[remote inventory] ssh {ssh_host} find ...", flush=True)
    proc = subprocess.run(
        ["ssh", ssh_host, remote_cmd],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("ssh inventory stderr:", proc.stderr, file=sys.stderr)
        raise SystemExit(f"ssh inventory failed (rc={proc.returncode})")

    inv = defaultdict(dict)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        fname = Path(line).name
        kind = kind_of(fname)
        base = basename_of(fname)
        if kind is None or base is None:
            continue
        # If duplicates exist (multiple suffix variants), keep the first.
        inv[base].setdefault(kind, line)
    print(f"[remote inventory] {len(inv)} basenames present at NERSC")
    return inv


def local_inventory(local_dir, basenames):
    """For each basename, return dict[kind] -> local Path or None."""
    inv = {}
    for base in basenames:
        per_kind = {}
        for kind, glob_template in LOCAL_GLOB_FOR_KIND.items():
            matches = sorted(local_dir.glob(glob_template.format(base=base)))
            per_kind[kind] = matches[0] if matches else None
        inv[base] = per_kind
    return inv


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("list_file", help="Path to chip *.list file (e.g. SE_C.list)")
    ap.add_argument("--local-dir", default="/Users/nugent/claude/ls4/test",
                    help="Local directory holding the flat collection of trios.")
    ap.add_argument("--ssh-host", default="perlmutter.nersc.gov")
    ap.add_argument("--remote-base",
                    default="/global/homes/n/nugent/ls4/archive-ls4/base/ls4")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would happen without doing it.")
    ap.add_argument("--no-delete", action="store_true",
                    help="Skip removal of orphaned local files.")
    args = ap.parse_args()

    list_file = Path(args.list_file).resolve()
    local_dir = Path(args.local_dir).resolve()
    if not list_file.is_file():
        raise SystemExit(f"missing list file: {list_file}")
    if not local_dir.is_dir():
        raise SystemExit(f"missing local dir: {local_dir}")

    chip = infer_chip(list_file)
    print(f"list file:    {list_file}")
    print(f"local dir:    {local_dir}")
    print(f"ssh host:     {args.ssh_host}")
    print(f"remote base:  {args.remote_base}")
    print(f"chip:         {chip or '(unknown)'}")
    if args.dry_run:
        print("MODE:         DRY RUN (no changes)")
    print()

    entries = list(parse_list(list_file))
    basenames = [b for _, _, b in entries]
    print(f"[list] {len(entries)} entries")

    remote = remote_inventory(args.ssh_host, args.remote_base, chip)
    local = local_inventory(local_dir, basenames)

    # ---- classify -----------------------------------------------------
    pull_list = []        # list of (kind, full_remote_path)
    delete_list = []      # list of local Path
    n_already_complete = 0
    n_unfetchable_no_local = 0
    pulled_per_kind = defaultdict(int)
    for night, time, base in entries:
        loc = local.get(base, {kind: None for kind in KINDS})
        rem = remote.get(base, {})

        have = {kind: bool(loc.get(kind)) or bool(rem.get(kind)) for kind in KINDS}
        achievable = all(have.values())

        if achievable:
            missing_local = [kind for kind in KINDS if not loc.get(kind)]
            if not missing_local:
                n_already_complete += 1
            else:
                for kind in missing_local:
                    pull_list.append((kind, rem[kind]))
                    pulled_per_kind[kind] += 1
        else:
            present_local = [kind for kind in KINDS if loc.get(kind)]
            if present_local:
                for kind in present_local:
                    delete_list.append(loc[kind])
            else:
                n_unfetchable_no_local += 1

    # ---- report plan --------------------------------------------------
    print()
    print(f"[plan] already complete (3/3 local): {n_already_complete}")
    print(f"[plan] to pull: {len(pull_list)} files "
          f"(image={pulled_per_kind[KIND_IMAGE]}, "
          f"sources={pulled_per_kind[KIND_SOURCES]}, "
          f"wcs={pulled_per_kind[KIND_WCS]})")
    print(f"[plan] orphans to delete: {len(delete_list)} files")
    print(f"[plan] unfetchable (nothing local, missing at NERSC): "
          f"{n_unfetchable_no_local}")

    if args.dry_run:
        if pull_list:
            print("\n[dry-run] WOULD pull:")
            for _, p in pull_list[:20]:
                print(f"  {p}")
            if len(pull_list) > 20:
                print(f"  ... ({len(pull_list)-20} more)")
        if delete_list:
            print("\n[dry-run] WOULD delete:")
            for p in delete_list[:20]:
                print(f"  {p}")
            if len(delete_list) > 20:
                print(f"  ... ({len(delete_list)-20} more)")
        print("\n[dry-run] no changes made.")
        return 0

    # ---- pull (one ssh+tar pipe) --------------------------------------
    if pull_list:
        # tar wants paths relative to remote_base for clean --transform.
        rel_paths = []
        for _, full in pull_list:
            try:
                rel = str(Path(full).relative_to(args.remote_base))
            except ValueError:
                rel = full  # fall back to absolute
            rel_paths.append(rel)
        # Feed the file list to remote tar via stdin (`tar -T -`) instead of
        # passing every path on the command line.  Inline argv hits the
        # ssh-command-line / argv length limit at ~hundreds of files; stdin
        # has no such limit.  We need a thread to write stdin while the
        # parent reads stdout, otherwise we can deadlock.
        import threading
        remote_tar = (
            f"cd {shlex.quote(args.remote_base)} && "
            f"tar cf - --transform='s|.*/||' -T -"
        )
        print(f"\n[pull] streaming {len(pull_list)} files via ssh+tar (stdin)...",
              flush=True)
        ssh_proc = subprocess.Popen(
            ["ssh", args.ssh_host, remote_tar],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tar_proc = subprocess.Popen(
            ["tar", "xf", "-", "-C", str(local_dir)],
            stdin=ssh_proc.stdout,
            stderr=subprocess.PIPE,
        )
        ssh_proc.stdout.close()  # let tar see EOF when ssh closes its stdout

        # Feed file list to ssh stdin in a separate thread so we don't block.
        filelist_blob = ("\n".join(rel_paths) + "\n").encode("utf-8")

        def _writer():
            try:
                ssh_proc.stdin.write(filelist_blob)
            finally:
                try:
                    ssh_proc.stdin.close()
                except BrokenPipeError:
                    pass
        writer_thread = threading.Thread(target=_writer)
        writer_thread.start()

        tar_stderr = tar_proc.communicate()[1]
        writer_thread.join()
        ssh_rc = ssh_proc.wait()
        ssh_stderr = ssh_proc.stderr.read()
        if ssh_rc != 0:
            print(f"[pull] WARN ssh rc={ssh_rc}", file=sys.stderr)
            if ssh_stderr:
                print(ssh_stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        if tar_proc.returncode != 0:
            print(f"[pull] WARN tar rc={tar_proc.returncode}", file=sys.stderr)
            if tar_stderr:
                print(tar_stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        print("[pull] done")

    # ---- delete orphans -----------------------------------------------
    if delete_list and not args.no_delete:
        print(f"\n[delete] removing {len(delete_list)} orphan files...")
        for p in delete_list:
            try:
                os.remove(p)
                print(f"  removed {p.name}")
            except OSError as e:
                print(f"  WARN cannot remove {p}: {e}", file=sys.stderr)
    elif delete_list:
        print(f"\n[delete] {len(delete_list)} orphans skipped (--no-delete)")

    # ---- final verification -------------------------------------------
    final_local = local_inventory(local_dir, basenames)
    final_complete = sum(
        1 for base in basenames
        if all(final_local[base].get(kind) for kind in KINDS)
    )
    print()
    print("=" * 60)
    print(f"  list:                            {len(entries)} entries")
    print(f"  already complete (start):        {n_already_complete}")
    print(f"  pulled:                          {len(pull_list)} files "
          f"(image={pulled_per_kind[KIND_IMAGE]}, "
          f"sources={pulled_per_kind[KIND_SOURCES]}, "
          f"wcs={pulled_per_kind[KIND_WCS]})")
    print(f"  orphans deleted:                 "
          f"{0 if args.no_delete else len(delete_list)}")
    print(f"  unfetchable / never-complete:    {n_unfetchable_no_local}")
    print(f"  final complete trios locally:    {final_complete}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
