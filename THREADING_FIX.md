# Threading Fix for Linux Visualization

## Problem

When running the visualization on Linux, the following panic occurred:

```
thread '<unnamed>' panicked at winit-0.30.12/src/platform_impl/linux/mod.rs:725:13:
Initializing the event loop outside of the main thread is a significant cross-platform compatibility hazard.
```

## Root Cause

By default, winit (the windowing library used by egui/eframe) requires the event loop to be created on the main thread for cross-platform compatibility. Our architecture runs:
- **Main thread**: Training loop
- **Separate thread**: Visualization (egui/eframe)

This violates winit's default threading requirements on Linux.

## Solution

Use winit's platform-specific extensions to allow event loop creation on any thread:

1. **Added winit dependency** to Cargo.toml:
   ```toml
   winit = "0.30"
   ```

2. **Used `with_any_thread(true)`** in the event loop builder:
   ```rust
   use winit::platform::x11::EventLoopBuilderExtX11;
   use winit::platform::wayland::EventLoopBuilderExtWayland;

   let options = eframe::NativeOptions {
       event_loop_builder: Some(Box::new(|builder| {
           #[cfg(target_os = "linux")]
           {
               if std::env::var("WAYLAND_DISPLAY").is_ok() {
                   builder.with_any_thread(true);  // Wayland
               } else {
                   builder.with_any_thread(true);  // X11
               }
           }
       })),
       // ... other options
   };
   ```

3. **Detection logic**: The fix detects whether the system is running Wayland or X11 and applies the appropriate platform extension.

## Why This Works

The `with_any_thread()` method explicitly tells winit that we understand the cross-platform implications and want to allow event loop creation on non-main threads. This is safe in our use case because:

1. We only create one visualization window
2. The visualization thread is dedicated to GUI operations
3. We use proper synchronization (`Arc<Mutex<>>`) for shared state
4. The training loop and visualization are properly decoupled

## Alternative Approaches Considered

### 1. Reverse Threading Model (Not Chosen)
Run GUI on main thread, training on worker thread:
- **Pros**: More conventional, no platform-specific code needed
- **Cons**: Requires major refactoring of main.rs, less intuitive API

### 2. Single-threaded with Polling (Not Chosen)
Run both training and GUI on same thread with periodic polling:
- **Pros**: No threading issues
- **Cons**: GUI updates would block training, poor performance

## Platform Support

This fix is specific to Linux (X11 and Wayland). Other platforms:
- **Windows**: Would also benefit from this fix if running in separate thread
- **macOS**: Has similar restrictions, would need similar platform-specific code
- **Web (WASM)**: N/A - single-threaded by nature

## Testing

Verified on:
- Linux with X11
- The program runs without panics
- Visualization window opens correctly
- Training continues in parallel with visualization

## References

- Winit documentation: https://docs.rs/winit/latest/winit/
- Platform-specific extensions: https://docs.rs/winit/latest/winit/platform/index.html
- eframe threading model: https://docs.rs/eframe/latest/eframe/
