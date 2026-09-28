from setuptools import find_packages, setup

package_name = "pick_place_common"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hojin",
    maintainer_email="hojinsong95@gmail.com",
    description=(
        "Method-agnostic half of the container pick-and-place task: MuJoCo "
        "plant node, shared scene/sensing constants, and the scene models."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_sim_node = pick_place_common.mujoco_sim_node:main",
        ],
    },
)
