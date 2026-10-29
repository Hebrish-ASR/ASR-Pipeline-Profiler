#!/usr/bin/env bash
# Build an Ubuntu .deb:  ./build-deb.sh  ->  dist/asrprof_<version>-1_amd64.deb
#
# /usr/lib/asrprof/ holds the app plus sherpa-onnx (+ sherpa-onnx-core) and the onnxruntime
# Python package, for the Hebrish example pipeline, installed with `uv pip install --target`
# (Ubuntu ships neither). PySide6, NumPy and sounddevice come from apt. The wheels are CPython-ABI
# specific, so the package is amd64 and pinned to the python3 minor it was built against.
# Same layout as the Hebrish desktop .deb (Hebrish-ASR/Hebrish-App, desktop/build-deb.sh).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PKG=asrprof
VERSION=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' asrprof/__init__.py)
DEB_REVISION=${DEB_REVISION:-1}
ARCH=$(dpkg --print-architecture)
PY=/usr/bin/python3
PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PYNEXT=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1] + 1}")')
[ "$ARCH" = amd64 ] || { echo "only amd64 is supported (got $ARCH)" >&2; exit 1; }

# TMPDIR: /tmp can be short of quota for ~100 MB of wheels; default to build/ in the repo
mkdir -p build && BUILD=$(mktemp -d -p "${TMPDIR:-$PWD/build}")
trap 'rm -rf "$BUILD"' EXIT
ROOT="$BUILD/root"
LIB="$ROOT/usr/lib/$PKG"
mkdir -p "$ROOT/DEBIAN" "$LIB" "$ROOT/usr/bin" "$ROOT/usr/share/applications" \
         "$ROOT/usr/share/icons/hicolor/scalable/apps" "$ROOT/usr/share/doc/$PKG" "$ROOT/usr/share/metainfo"

uv pip install --quiet --python "$PY" --target "$LIB" --no-deps "sherpa-onnx>=1.12" sherpa-onnx-core
uv pip install --quiet --python "$PY" --target "$LIB" --no-deps "onnxruntime>=1.17"
# Unused here (the Python binding links ONNX Runtime statically; Hebrish checked a session runs
# without the C library, 2026-10-24): the C library and the model-conversion tools.
rm -rf "$LIB"/onnxruntime/capi/libonnxruntime.so.* "$LIB"/onnxruntime/{tools,transformers,quantization}
uv pip install --quiet --python "$PY" --target "$LIB" --no-deps .
rm -rf "$LIB/bin" "$LIB/.lock" "$LIB"/*.dist-info/{uv_build.json,uv_cache.json,direct_url.json}
find "$LIB" -name '__pycache__' -type d -prune -exec rm -rf {} +
bundled=$(cd "$LIB" && ls -d sherpa*.dist-info onnxruntime*.dist-info | sed 's/\.dist-info$//' | tr '\n' ' ')

install -m755 packaging/debian/$PKG packaging/debian/$PKG-gui "$ROOT/usr/bin/"
install -m644 packaging/$PKG.desktop "$ROOT/usr/share/applications/"
install -m644 asrprof/data/icons/asrprof.svg "$ROOT/usr/share/icons/hicolor/scalable/apps/"
sed "s|@RELEASES@|    <release version=\"$VERSION\" date=\"$(date -u +%F)\"/>|" \
    packaging/$PKG.metainfo.xml > "$ROOT/usr/share/metainfo/app.hebrish.AsrProf.metainfo.xml"
install -m644 README.md "$ROOT/usr/share/doc/$PKG/"
install -m644 LICENSE "$ROOT/usr/share/doc/$PKG/copyright"

cat > "$ROOT/DEBIAN/control" <<CTRL
Package: $PKG
Version: $VERSION-$DEB_REVISION
Architecture: $ARCH
Section: devel
Priority: optional
Installed-Size: $(du -sk "$ROOT/usr" | cut -f1)
Maintainer: Daniel Rosehill <info@hebrish.app>
Homepage: https://github.com/Hebrish-ASR/ASR-Pipeline-Profiler
Depends: python3 (>= $PYVER), python3 (<< $PYNEXT), python3-pyside6.qtwidgets, qt6-svg-plugins, python3-numpy, python3-sounddevice, libportaudio2
Recommends: ffmpeg
Suggests: hebrish, python3-pynvml
Description: Profiler for multi-model local speech-to-text pipelines
 Shows where the time and the compute go in a local ASR pipeline made of
 several models (voice activity detection, speech recognition, paragraph or
 punctuation models, post-processing): per-stage wall time, CPU in cores and
 busy threads, memory per model, GPU use, and the junctions between stages.
 Batch and real-time replay modes, built-in reading samples with word error
 rate and paragraph-break checks. Desktop app (asrprof-gui), command line
 (asrprof) and an HTTP server (asrprof serve).
 Bundles: $bundled
CTRL

mkdir -p dist
OUT="dist/${PKG}_${VERSION}-${DEB_REVISION}_${ARCH}.deb"
dpkg-deb --root-owner-group -Zxz --build "$ROOT" "$OUT" >/dev/null
echo "Built: $OUT ($(du -h "$OUT" | cut -f1)); bundled: $bundled"
