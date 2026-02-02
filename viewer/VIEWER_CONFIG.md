# OpenSeadragon Viewer Configuration

## Fixing the Elastic/Springy Drag Behavior

OpenSeadragon's default settings create an elastic, springy feeling when dragging that many users find annoying. The settings below make dragging feel more direct and responsive while keeping smooth momentum scrolling.

## Key Settings for Direct Control

```javascript
// Core animation settings - direct drag, smooth momentum
animationTime: 0.55,         // Default: 1.2 - Tighter but smooth
springStiffness: 13,         // Default: 6.5 - Stiffer spring for snappy drag
constrainDuringPan: false,   // Default: false - Don't constrain during drag
immediateRender: false,      // Default: false - Wait for tiles
blendTime: 0.15,            // Default: 0 - Quick fade to hide tile loading

// Gesture settings - natural momentum without elasticity
gestureSettingsMouse: {
    flickEnabled: true,      // Keep momentum scrolling
    flickMinSpeed: 180,      // Default: 120 - Higher threshold (less accidental)
    flickMomentum: 0.14,     // Default: 0.25 - Tighter, natural decay
    dragToPan: true,
    scrollToZoom: true,
    clickToZoom: true,
    dblClickToZoom: true,
    pinchToZoom: true
}
```

## Fine-Tuning Guide

### To make dragging even more direct:
- **Increase** `springStiffness` to 15-20 (even stiffer)
- **Keep** `animationTime` at 0.6 (for smooth momentum)
- **Set** `flickEnabled` to false (removes all momentum)

### To adjust momentum/throwing:
- **Increase** `flickMomentum` to 0.25 (longer throw distance)
- **Decrease** `flickMomentum` to 0.1 (even less throw)
- **Increase** `flickMinSpeed` to 250+ (much harder to trigger)
- **Decrease** `animationTime` to 0.4 (snappier momentum)

### For smoother overall feel:
- **Increase** `animationTime` to 0.8-1.0 (smoother transitions)
- **Decrease** `springStiffness` to 8-10 (slightly springier)
- **Increase** `blendTime` to 0.25 (smoother tile fades)

## Other Useful Settings

```javascript
// Zoom responsiveness
zoomPerScroll: 1.2,          // Amount of zoom per scroll notch
pixelsPerWheelLine: 60,      // Scroll sensitivity
zoomPerClick: 2.0,           // Zoom factor for click zoom

// Tile loading
visibilityRatio: 0.5,        // Load tiles when 50% visible (earlier loading)
maxZoomPixelRatio: 2,        // Prevent over-zooming past image resolution

// View limits
minZoomLevel: 0.5,           // How far out you can zoom
maxZoomLevel: 40,            // How far in you can zoom
minZoomImageRatio: 0.5       // Minimum image:viewport ratio
```

## Testing Different Configurations

To test different settings in the browser console:

```javascript
// Make it extremely direct (no animations, no momentum)
viewer.animationTime = 0;
viewer.springStiffness = 20;
viewer.gestureSettingsMouse.flickEnabled = false;

// Make it smooth but responsive
viewer.animationTime = 0.3;
viewer.springStiffness = 8;
viewer.gestureSettingsMouse.flickMomentum = 0.3;

// Reset to our optimized defaults
viewer.animationTime = 0.55;
viewer.springStiffness = 13;
viewer.gestureSettingsMouse.flickEnabled = true;
viewer.gestureSettingsMouse.flickMinSpeed = 180;
viewer.gestureSettingsMouse.flickMomentum = 0.14;
```

## Comparison with Default Settings

| Setting | Default | Our Config | Effect |
|---------|---------|------------|--------|
| animationTime | 1.2s | 0.55s | 2.2× faster, balanced momentum |
| springStiffness | 6.5 | 13 | 100% stiffer (very direct) |
| constrainDuringPan | false | false | Smooth edge panning |
| immediateRender | false | false | Wait for tiles to load |
| blendTime | 0 | 0.15s | Smooth tile fade-in |
| flickMinSpeed | 120 | 180 | 50% higher threshold |
| flickMomentum | 0.25 | 0.14 | 44% tighter decay |

## Platform Differences

- **Desktop**: Mouse dragging now feels direct and responsive
- **Touch devices**: Still has natural momentum for finger flicking
- **Trackpad**: Two-finger scrolling remains smooth

## Known Issues and Workarounds

1. **If dragging still feels elastic**: Increase `springStiffness` to 15-20
2. **If momentum is too strong**: Reduce `flickMomentum` to 0.15
3. **If zoom feels sluggish**: Increase `zoomPerScroll` to 1.5
4. **If tiles pop in too late**: Decrease `visibilityRatio` to 0.3

## References

- [OpenSeadragon Options Documentation](http://openseadragon.github.io/docs/OpenSeadragon.html#Options)
- [OpenSeadragon Gesture Settings](http://openseadragon.github.io/docs/OpenSeadragon.GestureSettings.html)