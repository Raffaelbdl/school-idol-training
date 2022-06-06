import pickle

import bpy
import numpy as np

armatureN = "armatureN"
outputfile = bpy.path.abspath("output")
scene = bpy.data.scenes["Scene"]


for obj in bpy.data.objects:
    obj.select_set(False)

bpy.data.objects[armatureN].select_set(True)
armature = bpy.data.objects[armatureN]


def export_bone(bone):
    return bone.name, bone.head, bone.rotation_quaternion


bones = {bone.name: {"pos": [], "rot": []} for bone in armature.pose.bones}
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    print("Frame " + str(f) + ": ")

    for bone in armature.pose.bones:
        name, pos, rot = export_bone(bone)
        bones[name]["pos"].append(armature.matrix_world @ pos)
        bones[name]["rot"].append(rot)

for bone in armature.pose.bones:
    bones[bone.name]["pos"] = np.array(bones[bone.name]["pos"])
    bones[bone.name]["rot"] = np.array(bones[bone.name]["rot"])

# output buffer to file
with open(outputfile, "wb") as f:
    pickle.dump(bones, f)
