# CTranslate2 ROCm setup for AMD RX 6600 (GFX1030)

This guide explains how we set up and use a ROCm-enabled CTranslate2 build for the AMD RX 6600 (GFX1030) with this repository’s CLI.

- Recommended: install the prebuilt ROCm CTranslate2 wheel from `out/`.
- Alternatives: build from source using our Dockerfile or natively.

## Prerequisites

- ROCm stack installed and working on the host (verify `rocminfo` shows `gfx1030`).
- Python via PDM installed for this repo: `pdm install`.
- Optional: `patchelf` (needed for the vendor step below).
- Optional: Docker (for containerized building).

## Option A — Use prebuilt ROCm wheel (recommended)

1) Ensure dependencies:

```bash
pdm install
```

2) Install the ROCm CTranslate2 wheel from `out/` into the active PDM venv:

```bash
# Installs the newest out/ctranslate2-*.whl
pdm run faster-whisper-rocm install-ctranslate2

# Or specify the wheel explicitly
pdm run faster-whisper-rocm install-ctranslate2 --wheel out/ctranslate2-3.23.0-cp310-cp310-linux_x86_64.whl
```

3) Optional: vendor the shared library and patch RPATH if your loader can’t find `libctranslate2.so.3`:

```bash
# Requires patchelf on the system
pdm run python -m faster_whisper_rocm.cli.prepare_ctranslate2_rocm \
  out/ctranslate2_root/lib/libctranslate2.so.3
```

4) Verify import:

```bash
pdm run python -c "import ctranslate2; print('OK', ctranslate2.__version__, 'from', ctranslate2.__file__)"
```

## Option B — Build from source (Docker; targets GFX1030)

We provide a Dockerfile that builds CTranslate2 with ROCm for `gfx1030` and exports artifacts to `out/`.

```bash
# Build the image
docker build -f docker_rocm/Dockerfile.rocm -t ct2-rocm-gfx1030 .

# Export artifacts (wheel + ctranslate2_root) to host ./out
mkdir -p out
docker run --rm -v "$(pwd)/out:/out" ct2-rocm-gfx1030
```

After running the container, you should have:
- `out/ctranslate2-<ver>-*.whl` (Python wheel)
- `out/ctranslate2_root/` (installed libs, e.g., `libctranslate2.so.3`)

Install the wheel into your environment:

```bash
# Pick the newest wheel or specify explicitly
pdm run faster-whisper-rocm install-ctranslate2
# or
pdm run faster-whisper-rocm install-ctranslate2 --wheel out/ctranslate2-3.23.0-*.whl
```

Notes:
- The image’s default command copies artifacts to `/out`. Mount your host `./out` to `/out` as shown above.
- You can use `out/ctranslate2_root/lib/libctranslate2.so.3` with the vendor script if needed.

## Option C — Native build from source (advanced)

1) Prepare environment:

- Ensure ROCm toolchain (`hipcc`) works on the host.
- Confirm your GPU is reported as `gfx1030` by `rocminfo`.

2) Build CTranslate2 (v3.23.0) with HIP targets for `gfx1030`:

```bash
git clone https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2
git checkout v3.23.0
git submodule update --init --recursive

# Optional: apply our ROCm patch
git apply ../docker_rocm/ct2_3.23.0_rocm.patch || true

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DGPU_RUNTIME=HIP \
      -DCMAKE_HIP_ARCHITECTURES="gfx1030" \
      -DAMDGPU_TARGETS="gfx1030" \
      -DWITH_MKL=OFF -DWITH_DNNL=ON -DWITH_OPENBLAS=ON \
      -DENABLE_CPU_DISPATCH=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/../../out/ctranslate2_root \
      ..
make -j"$(nproc)" install
```

3) Install Python bindings:

```bash
cd ../python
pip install -r install_requirements.txt
pip install .
```

4) Optional vendor step (same as above) if the dynamic loader cannot locate the library.

## Using the CLI with GPU

Example invocations (GPU via ROCm-enabled CTranslate2):

```bash
# Print model/device info
pdm run faster-whisper-rocm model-info \
  --model Systran/faster-whisper-medium \
  --device cuda --compute-type float16

# Transcribe with a progress bar and limited segments for quick testing
pdm run faster-whisper-rocm transcribe data/samples/test_long.wav \
  --model Systran/faster-whisper-medium \
  --device cuda --compute-type float16 \
  --vad-filter \
  --output-format plain \
  --max-segments 10
```

Notes:
- In this project, `--device cuda` is used for GPU even with a ROCm-enabled build of CTranslate2 (consistent with our CLI usage examples and tests).
- Progress display auto-disables when writing to stdout in non-interactive contexts.

## Troubleshooting

- Library not found (`libctranslate2.so.3`): run the vendor script to copy the shared lib into the installed package and patch RPATH with `patchelf`.
- No GPUs detected: verify ROCm install (`rocminfo`) and that the GPU is shown as `gfx1030`.
- Build issues: confirm HIP targets (`gfx1030`) are set; when using Docker, ensure the host kernel/driver and ROCm versions are compatible.

## References

- `docker_rocm/Dockerfile.rocm`, `docker_rocm/ct2_3.23.0_rocm.patch`
- `faster_whisper_rocm/cli/prepare_ctranslate2_rocm.py`
- `out/ctranslate2-*.whl`, `out/ctranslate2_root/`
