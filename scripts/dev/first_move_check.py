import time, numpy as np, rclpy, sys
from rclpy.node import Node
from sensor_msgs.msg import JointState
rclpy.init(); n=Node("qcheck"); buf=[]
n.create_subscription(JointState,"/sim/joint_states",lambda m:buf.append(np.linalg.norm(m.velocity)),10)
t0=time.time()
while time.time()-t0<float(sys.argv[1]): rclpy.spin_once(n,timeout_sec=0.1)
buf=np.array(buf[len(buf)//3:]) if buf else np.array([99.])
print("FLAKE" if buf.mean()>2 else "OK", round(float(buf.mean()),2))
