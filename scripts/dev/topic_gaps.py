import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
rclpy.init(); n=Node("gaps"); last={}
def mk(name,thresh):
    def cb(m):
        t=time.time()
        if name in last and t-last[name]>thresh: print(f"{t:.2f} {name} arrival gap {t-last[name]:.2f}",flush=True)
        last[name]=t
    return cb
n.create_subscription(Float64MultiArray,"/perception/camera_detections",mk("detections",0.3),qos_profile_sensor_data)
n.create_subscription(JointState,"/sim/joint_states",mk("joint_states",0.1),10)
n.create_subscription(Float64,"/mpc/speed_scale",mk("speed_scale",0.3),10)
t0=time.time()
while time.time()-t0<float(sys.argv[1]): rclpy.spin_once(n,timeout_sec=0.05)
