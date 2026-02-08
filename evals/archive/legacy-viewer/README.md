# WSI DeepZoom Viewer

This is a minimal OpenSeadragon viewer for DeepZoom pyramids.

## Usage

Serve the repo and open the viewer in a browser:

```
python3 -m http.server 8000
```

Then visit:

```
http://localhost:8000/viewer/?dzi=data/demo_out/baseline_pyramid.dzi
```

You can also paste a DZI path into the input box at the top.

## Server viewer

If the Rust tile server is running, open:

```
http://localhost:8000/viewer/server.html
```
