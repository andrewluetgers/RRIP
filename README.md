# RRIP
## Residual packfiles

Generate packfiles for faster server reads:

```
python cli/wsi_residual_tool.py pack --residuals data/demo_out/residuals_q32 --out data/demo_out/residual_packs
```

Run the server with packfiles:

```
TURBOJPEG_SOURCE=explicit \
TURBOJPEG_DYNAMIC=1 \
TURBOJPEG_LIB_DIR=$(brew --prefix jpeg-turbo)/lib \
TURBOJPEG_INCLUDE_DIR=$(brew --prefix jpeg-turbo)/include \
cargo run --manifest-path server/Cargo.toml -- \
  --slides-root data \
  --port 3008 \
  --residual-pack-dir data/demo_out/residual_packs
```
