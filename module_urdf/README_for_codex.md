# README for codex about URDF model of A-MSRR module
- holon.urdf.xacroは、ROS2経由のgazebo simulation用に書かれたURDFです。
- "root","pitch_connect_dummy_*", "yaw_connect_dummy_*"はURDFの仕様上必要であったdummy linkです。必要無い場合は無視して構いません。
- 結合は、pitch_dock_mechとyaw_dock_mechで行われます。pitch to pitchやyaw to yawはありません。
- pitch_connect_pointとyaw_connect_pointは結合機構合体時の結合点です。
- "fc"はflight controllerのリンクであり、この原点にIMUがあります。基本的に、ここをモジュールの機体座標系原点とします。