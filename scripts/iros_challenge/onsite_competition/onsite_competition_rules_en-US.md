
# Nav Track Onsite Competition Rules
English Version | [中文版](./onsite_competition_rules_zh-CN.md)

## 1. Task Description
This track focuses on building a multimodal mobile robot navigation system with language understanding capabilities. Participants must design a perception–decision pipeline that performs the full process of:
- Egocentric visual perception,
- Natural language instruction understanding,
- Historical trajectory modeling,
- Navigation action prediction.

The submitted algorithms will be deployed on a real robot, which must navigate indoors under natural language guidance. The robot should robustly handle camera shake, height changes, and local obstacle avoidance, ensuring safe and reliable vision-language navigation.

Key Challenges:
- Effectively fusing language and visual information to support an integrated perception–decision–control process.
- Operating robustly on a real robotic platform, handling viewpoint shake, height changes, and local obstacle avoidance during navigation.
- Generalizing to unseen indoor environments and novel language instructions for improved robustness and adaptability.

## 2. Competition Environment & Equipment
### 2.1 Competition Venue
A realistic apartment-like environment will be built, consisting of connected rooms (living room, bedroom, kitchen, corridor, bathroom, etc.) with typical household furniture and decorations.

### 2.2 Robot
The competition will use Robot (provided by the organizers) with same RGB-D camera and Sensor configuration. Detailed specifications and open-source navigation resources will be provided.
- Teams will have an on-site debugging session on October 18.
- Final code must be submitted 1 days before the competition.

## 3. Task Setup
The challenge centers on vision-language fusion for cross-room end-to-end navigation.
 The organizers will predefine ~10 natural language navigation instructions, each with a corresponding start and goal position.
- Each instruction must cross at least one room.
- Ground-truth path length will be between 5–20 meters.
- The goal location must be precise, unambiguous, and clearly defined.

## 4. Competition Rules
### 4.1 Pre-competition Preparation
- Teams must package the competition image in advance according to the GitHub documentation.
- A standardized debugging time and environment will be provided onsite. Teams may use model weights different from the online stage and make environment-specific configuration adjustments, but modifications to the core algorithm logic are strictly prohibited.

### 4.2 Procedure
Each team will receive 10 instructions. Start from any instruction.
For each instruction:
- Move the robot to the given starting position.
- Provide the instruction to the robot and raise your hand to signal the start.
- If execution fails, teams may retry or skip the instruction.
- Instruction can be repeated if failed (timeout, collision, human intervention, etc.).
- Before each attempt, the algorithm state must be reset and previous observations cleared.

### 4.3 Time Limit
Each instruction has a maximum runtime of 6 minutes.
The total maximum time per team is 55 minutes, including:
- Moving to start points,
- Discussion about retry/skip decisions,
- Executing instructions.
If time runs out mid-instruction, the robot’s position at timeout will be used for scoring, and remaining instructions will be considered skipped.

### 4.4 Fair Play
Participants must not seek unfair advantages. Restrictions include:
- No pre-mapping of the environment before the competition.
- No code changes during the run except for:
  - Modifying input instructions,
  - Fixing fatal runtime errors (crashes).
- The robot must be teleoperated only to reach the starting position (confirmed by referees).
- No human intervention is allowed during navigation.
- The submitted runtime container/image must be frozen and submitted before the event; no internet updates are allowed during competition.

### 4.5 Refereeing
- Each match is monitored by two referees remotely via the venue surveillance system.
- Referees remain outside the arena and observe in real time via cameras.
- All matches are recorded and live-streamed.
- Robot execution is controlled from a centralized control console by the organizers.
- Referees have remote emergency stop (E-Stop) authority and will intervene if unsafe or unfair behavior is detected.

## 5. Scoring System
### 5.1 Onsite Scoring
The competition will calculate each team’s success rate over ten tasks, and the final score will be obtained by taking a weighted average with the success rate from the online competition.

**Scoring Rules**:
Successfully completing one instruction will be one success, and the completion time for that instruction will be recorded.

| Action | Score Impact |
|:--|:--|
| Successfully reach goal | success |
| Minor scrape or Collision with obstacle | fail |

If there is a trend of continuous collisions, the referee has the right to terminate the robot’s current action, with the severity of the impact determined by the on-site referee.

**Success Condition**:
 The goal is defined as a 2m-radius circular area (no wall crossing). The run is considered successful if the robot stops inside this area.

**Ranking Rules (Onsite Competition)**:
- The final score is the success rate, calculated as the number of successful instructions divided by the total number of instructions.

## 5.2 Final Results
Final results combine online phase and onsite phase scores:
- Final Score Calculation:
Final Score = (Online SR × 40%) + (Onsite SR × 60%)
