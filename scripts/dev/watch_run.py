import sys, time, csv, numpy as np, rclpy, pinocchio as pin
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from pick_place_mpc.obstacle_supervisor_node import MJCF_PATH, ARM_POINTS
dur=float(sys.argv[1]); out=sys.argv[2]
rclpy.init(); n=Node("watch3")
model=pin.buildModelFromMJCF(MJCF_PATH); data=model.createData(); fids=[(model.getFrameId(a),r) for a,r in ARM_POINTS]
st=dict(holding=0,hold=0,scale=1.0,truth=None,goal=None,goal_t=None,gv=0.0,q=None,qd=None,events=[])
def js(m): st["q"]=np.array(m.position); st["qd"]=np.array(m.velocity)
def gl(m):
    g=np.array(m.data[:3]); t=time.time()
    if st["goal"] is not None: st["gv"]=np.linalg.norm(g-st["goal"])/max(t-st["goal_t"],1e-3)
    st["goal"],st["goal_t"]=g,t
n.create_subscription(JointState,"/sim/joint_states",js,10)
n.create_subscription(Bool,"/mpc/hold",lambda m:st.__setitem__("hold",int(m.data)),10)
n.create_subscription(Float64,"/mpc/speed_scale",lambda m:st.__setitem__("scale",m.data),10)
n.create_subscription(Float64MultiArray,"/env/dynamic_obstacle",lambda m:st.__setitem__("truth",np.array(m.data)),10)
n.create_subscription(Float64MultiArray,"/mpc/goal",gl,10)
def act(m):
    st["events"].append((time.time()-t0,m.data)); st["holding"]=1 if m.data.startswith("pick_at") else 0
n.create_subscription(String,"/task/action",act,10)
w=csv.writer(open(out,"w")); w.writerow(["t","hold","scale","qdot","goal_speed","true_gap","ee_x","ee_y","ee_z","tx","ty","pt","tilt","yaw","holding"])
t0=time.time(); nxt=t0
while time.time()-t0<dur:
    rclpy.spin_once(n,timeout_sec=0.02)
    if time.time()<nxt or st["q"] is None: continue
    nxt+=0.05
    pin.forwardKinematics(model,data,st["q"]); pin.updateFramePlacements(model,data)
    ee=np.array(data.oMf[fids[-1][0]].translation)
    R=np.array(data.oMf[fids[-1][0]].rotation)
    tilt=float(np.degrees(np.arccos(np.clip(-R[:,2][2],-1,1))))
    yaw=float(np.degrees(np.arctan2(R[1,0],R[0,0])))
    gap=np.nan; pt=""
    if st["truth"] is not None and st["truth"][3]>0:
        T=st["truth"]; top=T[5] if len(T)>5 else T[2]+T[3]; rad=T[3]
        def cg(p):
            if len(T)>6 and T[6]==1:  # person: column floor..top as capsule
                c=np.array([T[0],T[1],np.clip(p[2],0.0,max(top-rad,0.0))]); return np.linalg.norm(p-c)-rad
            return np.linalg.norm(T[:3]-p)-rad
        gs=[cg(np.array(data.oMf[f].translation))-r for f,r in fids]
        k=int(np.argmin(gs)); gap=gs[k]; pt=ARM_POINTS[k][0]
    w.writerow([round(time.time()-t0,2),st["hold"],round(st["scale"],2),round(float(np.linalg.norm(st["qd"])),3),round(st["gv"],3),round(gap,3),*ee.round(3),round(st["truth"][0],3) if st["truth"] is not None else "",round(st["truth"][1],3) if st["truth"] is not None else "",pt,round(tilt,2),round(yaw,1),st["holding"]])
print("events",[(round(a,1),b[:30]) for a,b in st["events"]])
