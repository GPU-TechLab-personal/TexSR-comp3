"""Point a grouped TFRecord's meta at the pair folder it now sits beside.

texsr_build_grouped_tfrecord.py writes the pair root it read into
<record>.meta.json ("root"), and texsr_verify_grouped.py fails when that string
differs from the root it is asked to verify against. Copying the pairs and the
record to the data drive (scripts/data.sh import) changes the path but not one
byte of either, so the old string makes an otherwise clean record FAIL.

This rewrites "root" to the new folder and keeps the old value in
"root_built_from". It changes nothing else: the record file, its sha256, bytes
and record_count stay as they are. It is safe because the verifier that runs
next compares every one of the record's PNG strings, byte for byte, against the
files under the new root - a record that does not match them still FAILs.

  python variants/util/rebase_grouped_root.py --tfrecord <record> --root <pairs>
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord", required=True)
    ap.add_argument("--root", required=True, help="the pair folder the record now matches")
    args = ap.parse_args()

    meta_path = args.tfrecord + ".meta.json"
    if not os.path.isfile(meta_path):
        sys.exit("no %s" % meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    old = meta.get("root")
    new = os.path.normpath(os.path.abspath(args.root))
    if old is not None and os.path.normpath(str(old)) == new:
        print("[rebase] meta root already %s" % new)
        return
    meta.setdefault("root_built_from", old)
    meta["root"] = new
    tmp = meta_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, meta_path)
    print("[rebase] meta root %s -> %s (old value kept as root_built_from)" % (old, new))


if __name__ == "__main__":
    main()
