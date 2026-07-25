# Order 9 native posture IK

`order9_posture_native.cpp` is the host-compiled C++17/Eigen/FCL kernel used
by `Order9PostureTrajectoryResolver`.

Build it once on each target architecture:

```bash
scripts/build_order9_posture_native.sh
```

The extension is written below `build/order9_posture_native/`, which is
intentionally ignored because CPython ABI and CPU architecture are
platform-specific.  Set `AMSRR_ORDER9_POSTURE_NATIVE_PATH` only when a
deployment keeps the extension at another explicit path.

The ordinary resolver can fall back to its readable Python IK when the
extension is absent.  The production convex collision gate cannot: it
requires the native extension and fails closed.  Runtime code never invokes
the compiler.
