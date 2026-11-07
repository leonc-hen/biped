Phase 0 Reading Notes
Draft notes for Step 4 (0.7), 7 November 2026. Written from web research. Each note answers the question set for that source. Items marked TO CONFIRM are things I could not verify from what I found, so check them while you skim the source and then delete the marker.
1. Upkie
Link: https://github.com/upkie/upkie (build instructions in its wiki, CAD in https://github.com/upkie/parts)
Question: how do they structure the control stack and the sim to real interface?
Answer: Upkie is an open source self balancing wheeled biped. Wheels do the balancing and legs handle uneven terrain. It is meant to be built at home from parts ordered online, including mjbots actuators. The software is split into spines and agents. A spine talks to either the simulation or the real actuators. An agent is the program that implements a behavior. Because the agent only sees the spine, the same agent code runs in sim and on the robot. The real robot runs motion control onboard a Raspberry Pi with an mjbots pi3hat. Agents are written against Gymnasium environments (upkie.envs).
Key numbers: the README example runs an environment at 200 Hz. Its simplest balancer is a proportional feedback law that multiplies the pitch observation by 10 to get the wheel command. Hardware and software are Apache 2.0 licensed.
What I would copy: the spine and agent split, so my policy and LQR code never know whether they are in sim or on hardware. Also the idea of starting with a trivial pitch feedback balancer as a baseline.
What I would not copy: single environment (non vectorized) training. Their environments run one robot at a time so they work as is on hardware. My Phase 2 training needs thousands of parallel environments on the GPU, so I need a separate vectorized training environment that shares the same model and parameters.
Open questions: how do they handle the control rate and latency between spine and agent? TO CONFIRM in the introduction to spines. What do their PID, Pink and PPO balancers use as observations and actions?
2. Woofer and Pupper (Stanford)
Links: Woofer and Pupper short paper (https://www.seas.upenn.edu/~posa/DynamicWalking2020/695-1037-1-RV.pdf) and the Pupper benchmark paper (https://arxiv.org/abs/2110.00736)
Question: what did cheap actuators limit?
Answer: The original Pupper used hobby servos. Servos rule out torque control because of backlash and high reflected inertia, so the robot was limited to open loop strategies based on Raibert style heuristics. The later Pupper uses torque controllable brushless motors (M2006 gearmotor with a C610 controller), which allows impedance and torque based learning and optimization approaches, and it can be built in under 8 hours.
Key numbers: Woofer is about 8 kg and costs about $6000 to build. The newer Pupper is 2.1 kg, 12 degrees of freedom, and under $2000. For comparison in their table: Solo 2.2 kg at $4700, D'Kitty $4200 with no torque control, Mini Cheetah about 9 kg.
What I would copy: the idea of defining benchmark tasks with numbers (their Sprint and Scramble) so results are comparable. This matches my requirements table. Also their reproducibility mindset: documented so others can rebuild it.
What I would not copy: servo level actuation. My CubeMars style actuators are torque controllable, so the sim must model the actuator (current limit, reflected inertia, friction, and backlash if present) instead of assuming ideal torque.
Note on the Phase 0 document: it said Pupper costs under $600 and uses mainly servos. That matches the early servo version only. The newer benchmark version is brushless and costs about $2000.
Open questions: how large was the sim to real gap because of backlash? TO CONFIRM in the paper.
3. StackForce mini wheeled legged robot
Links: https://wiki.seeedstudio.com/stackforce_mini_wheeled_legged_robot, course code at https://gitee.com/StackForce/bipedal_wheeled_robot, theory page in the Seeed AI Robotics Academy repo on GitHub
Question: what is the leg layout and what balance controller do they use?
Answer: The robot is built on StackForce's controller, motor driver and servo driver boards. Parts list: two 2208 gimbal motors with MT6701 encoders drive the wheels, four DS041MG servos drive the legs (so two servos per leg), 12 V LiPo battery, 29 printed parts. Motor control uses field oriented control with speed, position and torque modes. The software includes a whole body kinematic model and a state adaptive self balancing algorithm. The Seeed course page lists its wheel legged course as covering balance control and LQR motion planning, so the wheel balance controller is LQR based.
Key numbers: pre sale price around $578. Control rates and gains: TO CONFIRM in the code.
What I would copy: the idea of a kinematic model that converts a desired leg length into servo angles, with LQR balancing underneath.
What I would not copy: hobby servos for the legs and gimbal motors for the wheels. They are too weak for jumping.
Open questions: what is the exact leg linkage (for example a five bar)? How does the LQR gain change with leg length? TO CONFIRM in the course code and theory page. The code is on Gitee, which may need a login, so use the zip download.
4. MuJoCo documentation
Link: https://mujoco.readthedocs.io
Question: how do I model motors, contacts and a closed linkage leg correctly?
This note is a reading checklist written from my background knowledge, not from the docs. Confirm each item while reading and replace it with what the docs actually say.
Modeling: TO CONFIRM the MJCF basics, the default timestep, and how to set the timestep to 0.002 s (500 Hz) in the option tag.
Actuators: TO CONFIRM position actuators (kp and kv), the gear ratio, ctrlrange and forcerange limits, and how to approximate a DC motor torque speed curve.
Joints: TO CONFIRM armature (reflected rotor inertia), damping, frictionloss for gearbox friction.
Contacts: TO CONFIRM the friction parameters (sliding, torsional, rolling) and solref and solimp for tire compliance.
Equality constraints: TO CONFIRM connect and weld constraints, which are the tool for a closed loop linkage leg.
Key numbers: record the defaults you rely on.
5. MuJoCo Playground
Links: paper https://arxiv.org/abs/2502.08844, site https://playground.mujoco.org, repo https://github.com/google-deepmind/mujoco_playground
Question: how are environments structured, and how is domain randomization done?
Answer: Playground is an open source framework built on MJX (the JAX based, GPU capable MuJoCo) for simulation, training and sim to real transfer. Its environments fall into three categories: DeepMind Control Suite, locomotion and manipulation. Locomotion covers several quadrupeds and humanoids. Environments are loaded by name through a registry, for example locomotion.load(env_name, config=env_cfg). They derive from a common MjxEnv base class. Training typically wraps an environment with BraxEnvWrapper and calls the Brax PPO train function, which takes a randomization function argument. That argument is where domain randomization plugs in. The authors describe the design as minimal abstraction with monolithic environment definitions that are easy to hack, and their follow up framework mjlab uses a more modular manager based design.
Key numbers: the paper reports policies training in minutes on a single GPU. For your 5060 (8 GB), expect to lower the number of parallel environments if you hit memory errors.
What I would copy: the structure of one locomotion environment (reset, step, reward terms, randomization function) as the template for my wheeled biped.
What I would not copy: anything that assumes a quadruped or humanoid model.
Open questions: which file in the locomotion folder is the closest template for a small robot? Where exactly is the randomization function defined for the Go1 environment? TO CONFIRM in the repo.
Papers (verified)
1. Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26), eaau5872, 2019. arXiv:1901.08652. A policy is trained in simulation and transferred to the ANYmal quadruped. The actuator network idea is from my memory, TO CONFIRM when reading.
2. Rudin, Hoeller, Reist, Hutter. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL 2021, PMLR 164:91-100, 2022. arXiv:2109.11978. Thousands of simulated robots on one GPU, flat terrain policies in under four minutes.
3. Peng, Andrychowicz, Zaremba, Abbeel. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA 2018. arXiv:1710.06537. Randomizing simulator dynamics lets the policy adapt to real dynamics. The demonstration is a robot arm pushing task, not a legged robot.
4. Klemm, Morra, Gulich, Mannhart, Rohr, Kamel, de Viragh, Siegwart. LQR-Assisted Whole-Body Control of a Wheeled Bipedal Robot with Kinematic Loops. IEEE RA-L 5(2), 3745 to 3752, 2020. arXiv:2005.11431. A hierarchical whole body controller with an LQR included as a motion task to handle the non minimum phase balancing dynamics, and lean angle regulation using the zero moment point.
5. Klemm et al. Ascento: A Two-Wheeled Jumping Robot. ICRA. arXiv:2005.11435. A compact wheeled biped that moves fast on flat ground and jumps over obstacles. Year not confirmed, TO CONFIRM on the page.
6. Raibert. Legged Robots That Balance (book, 1986). Not searched. TO CONFIRM title and year before citing.
Ideas I will steal
1. Agent and spine split so the same controller code runs in sim and on hardware (Upkie). Goes in sim/ and firmware/ interfaces.
2. Start with a trivial pitch feedback balancer as a baseline before LQR (Upkie). Goes in control/.
3. Define benchmark tasks with numbers so progress is measurable (Pupper). Goes in requirements.md.
4. Model the actuator, including reflected inertia and friction, because cheap or backlashy drives limit torque control (Pupper). Goes in the Level 1 model.
5. Kinematic model for leg length plus LQR balancing underneath (StackForce, Ascento). Goes in Phase 1 and 2.
6. Use the Playground randomization function slot for dynamics randomization (Playground, Peng). Goes in sim/ Level 3.
7. Include LQR as a task inside a larger whole body controller if recovery needs it (Klemm). Goes in decisions.md as a fallback.
8. Small variable length inverted pendulum view of a wheel legged robot, with LQR for balance and PD or virtual model control for body height, appears in a separate open access paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC12431389). Worth reading in Phase 1.